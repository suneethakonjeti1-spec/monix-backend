require('dotenv').config();
const express = require('express');
const multer = require('multer');
const PDFParser = require('pdf2json');
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');
const admin = require('firebase-admin');

// ─── App Setup ────────────────────────────────────────────────────────────────

const app = express();

app.use(cors({
    origin: process.env.ALLOWED_ORIGIN || '*',
    methods: ['GET', 'POST'],
}));

app.use(express.json({ limit: '10mb' }));

// ─── Firebase Admin Init ──────────────────────────────────────────────────────
// Requires GOOGLE_APPLICATION_CREDENTIALS env var pointing to your service account JSON
// OR set FIREBASE_SERVICE_ACCOUNT env var with the JSON string directly.

let adminInitialized = false;
try {
    const serviceAccount = process.env.FIREBASE_SERVICE_ACCOUNT
        ? JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT)
        : undefined;

    admin.initializeApp({
        credential: serviceAccount
            ? admin.credential.cert(serviceAccount)
            : admin.credential.applicationDefault(),
    });
    adminInitialized = true;
    console.log('✅ Firebase Admin SDK initialized.');
} catch (err) {
    console.warn('⚠️  Firebase Admin SDK not initialized. Auth middleware will be skipped.');
    console.warn('   Set FIREBASE_SERVICE_ACCOUNT env var to enable authentication.');
}

// ─── Multer ───────────────────────────────────────────────────────────────────

const MAX_FILE_SIZE_MB = 15;
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: MAX_FILE_SIZE_MB * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
        if (file.mimetype !== 'application/pdf') {
            return cb(new Error('Only PDF files are accepted.'), false);
        }
        cb(null, true);
    },
});

// ─── Gemini AI ────────────────────────────────────────────────────────────────

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ─── Auth Middleware ──────────────────────────────────────────────────────────

async function requireAuth(req, res, next) {
    if (!adminInitialized) return next(); // Dev mode: skip auth if admin not configured

    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Unauthorized: missing token.' });
    }

    const idToken = authHeader.split('Bearer ')[1];
    try {
        const decoded = await admin.auth().verifyIdToken(idToken);
        req.user = decoded; // uid, email, etc. available downstream
        next();
    } catch {
        return res.status(403).json({ error: 'Forbidden: invalid or expired token.' });
    }
}

// ─── PDF Extraction ───────────────────────────────────────────────────────────

function extractTextFromPDF(pdfBuffer) {
    return new Promise((resolve, reject) => {
        const pdfParser = new PDFParser(null, 1);
        pdfParser.on('pdfParser_dataError', (errData) => reject(new Error(errData.parserError)));
        pdfParser.on('pdfParser_dataReady', () => resolve(pdfParser.getRawTextContent()));
        pdfParser.parseBuffer(pdfBuffer);
    });
}

// ─── Safe JSON Parse ──────────────────────────────────────────────────────────

function safeParseAIJson(rawText) {
    // Strip markdown code fences
    let cleaned = rawText.replace(/```json/gi, '').replace(/```/g, '').trim();

    // If it looks like a truncated array, attempt to close it cleanly
    if (cleaned.startsWith('[') && !cleaned.endsWith(']')) {
        const lastClose = cleaned.lastIndexOf('}');
        if (lastClose !== -1) {
            cleaned = cleaned.substring(0, lastClose + 1) + ']';
        } else {
            cleaned += ']';
        }
    }

    return JSON.parse(cleaned); // Let the caller handle parse errors
}

// ─── ROUTE 1: Upload & Analyze PDF ───────────────────────────────────────────

app.post('/upload', requireAuth, upload.single('pdfFile'), async (req, res) => {
    try {
        console.log('\n──────────────────────────────────────');
        console.log(`📥 Upload request | user: ${req.user?.uid ?? 'dev-mode'}`);

        if (!req.file) {
            return res.status(400).json({ error: 'No valid PDF file received.' });
        }

        console.log(`📄 File: ${req.file.originalname} (${(req.file.size / 1024).toFixed(1)} KB)`);
        console.log('🔍 Extracting text from PDF...');

        let extractedText;
        try {
            extractedText = await extractTextFromPDF(req.file.buffer);
        } catch (pdfErr) {
            console.error('PDF extraction failed:', pdfErr);
            return res.status(422).json({ error: 'Could not extract text from PDF. Is it a scanned image? Only text-based PDFs are supported.' });
        }

        if (!extractedText || extractedText.trim().length < 50) {
            return res.status(422).json({ error: 'PDF appears to be empty or contains no readable text.' });
        }

        console.log(`📝 Extracted ${extractedText.length} chars. Sending to Gemini...`);

        const prompt = `You are a financial AI specializing in Indian bank statements and UPI wallet statements (like PhonePe, GPay, Paytm).

Extract ALL transactions from the text below. Follow these rules strictly:

1. If an "Opening Balance" entry exists, include it as the very first transaction with a positive amount.
2. Only extract actual money movements — Debit (negative) and Credit (positive). Ignore "Balance" columns.
3. Normalize amounts: debits must be NEGATIVE numbers, credits must be POSITIVE numbers.
4. Assign EXACTLY one category from this fixed list:
   "Utilities" | "Salary" | "Food & Dining" | "Shopping" | "Auto & Transport" | "Entertainment" | "Transfers & Loans" | "Groceries" | "UPI Payment" | "Other"
5. Date format: use ISO 8601 (YYYY-MM-DD). If only day/month visible, infer the year from context.
6. The "description" field must be plain text only — no double quotes, no backslashes, no special characters.
7. Return ONLY a valid JSON array. No markdown, no explanation, no code fences. Each object must have exactly these keys: "date", "description", "amount", "category".

Statement text:
${extractedText}`;

        let aiResponse;
        try {
            aiResponse = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: [{ role: 'user', parts: [{ text: prompt }] }],
                config: { responseMimeType: 'application/json' },
            });
        } catch (aiErr) {
            console.error('Gemini API error:', aiErr);
            return res.status(502).json({ error: 'AI service is unavailable. Please try again later.' });
        }

        let parsed;
        try {
            parsed = safeParseAIJson(aiResponse.text);
        } catch (parseErr) {
            console.error('JSON parse failed. Raw AI output:\n', aiResponse.text?.slice(0, 500));
            return res.status(422).json({ error: 'AI returned malformed data. Please retry — this sometimes happens with complex PDFs.' });
        }

        // Validate and sanitize each transaction
        const validated = parsed
            .filter(t => t && typeof t.amount === 'number' && !isNaN(t.amount) && t.date && t.description)
            .map(t => ({
                date: String(t.date).trim(),
                description: String(t.description).replace(/[<>"'&]/g, '').trim().slice(0, 200),
                amount: parseFloat(t.amount.toFixed(2)),
                category: t.category || 'Other',
            }));

        console.log(`✅ Parsed ${validated.length} transactions. Sending to frontend.`);
        res.json(validated);

    } catch (error) {
        console.error('Unexpected server error in /upload:', error);
        res.status(500).json({ error: 'An unexpected error occurred. Please try again.' });
    }
});

// Handle multer file size error specifically
app.use((err, req, res, next) => {
    if (err.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({ error: `File too large. Maximum size is ${MAX_FILE_SIZE_MB}MB.` });
    }
    if (err.message === 'Only PDF files are accepted.') {
        return res.status(400).json({ error: err.message });
    }
    next(err);
});

// ─── ROUTE 2: AI Chat ─────────────────────────────────────────────────────────

app.post('/chat', requireAuth, async (req, res) => {
    try {
        const { question, history, roastMode = false } = req.body;

        if (!question || typeof question !== 'string' || question.trim().length === 0) {
            return res.status(400).json({ error: 'Question cannot be empty.' });
        }
        if (question.length > 1000) {
            return res.status(400).json({ error: 'Question is too long (max 1000 characters).' });
        }

        console.log(`💬 Chat | user: ${req.user?.uid ?? 'dev'} | roast: ${roastMode}`);

        const CATEGORIES = ['Utilities', 'Salary', 'Food & Dining', 'Shopping', 'Auto & Transport', 'Entertainment', 'Transfers & Loans', 'Groceries', 'UPI Payment', 'Other'];
        const totalSpend = (history || []).filter(t => t.amount < 0).reduce((s, t) => s + Math.abs(t.amount), 0);
        const totalIncome = (history || []).filter(t => t.amount > 0).reduce((s, t) => s + t.amount, 0);

        const systemPersona = roastMode
            ? `You are Monix in "Roast Mode" — a brutally honest, sarcastic, and slightly savage financial advisor. You point out bad spending habits with dark humor. Keep it short (2-4 sentences max). Never be genuinely mean or offensive. Always end with one actionable tip.`
            : `You are Monix, a professional and empathetic financial advisor specializing in Indian personal finance. Be concise, specific, and helpful. Reference actual transaction data when relevant. Use INR (₹). Keep responses under 150 words.`;

        const contextSummary = `
Total income: ₹${totalIncome.toFixed(2)}
Total spending: ₹${totalSpend.toFixed(2)}
Net savings: ₹${(totalIncome - totalSpend).toFixed(2)}
Transaction count: ${(history || []).length}
Recent 10 transactions: ${JSON.stringify((history || []).slice(-10))}`;

        const prompt = `${systemPersona}

Financial context:
${contextSummary}

User question: ${question}`;

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        res.json({ answer: response.text });
    } catch (error) {
        console.error('Chat error:', error);
        res.status(500).json({ error: 'Failed to get AI response. Please try again.' });
    }
});

// ─── ROUTE 3: Predictions ─────────────────────────────────────────────────────

app.post('/predict', requireAuth, async (req, res) => {
    try {
        const { history } = req.body;

        if (!history || !Array.isArray(history) || history.length < 5) {
            return res.json([]); // Not enough data to predict — return empty gracefully
        }

        console.log(`🔮 Predict | user: ${req.user?.uid ?? 'dev'} | ${history.length} txns`);

        const prompt = `Analyze this Indian bank transaction history and identify recurring monthly expenses.

Transactions: ${JSON.stringify(history)}

Rules:
1. Only include expenses that appear at least twice with similar amounts (±20% variance is acceptable).
2. Predict EXACTLY 3 bills the user will likely pay next month.
3. Use real merchant names or service types (e.g., "Netflix Subscription", "Electricity Bill").
4. Return ONLY a valid JSON array, no markdown, no explanation.
5. Each object must have exactly: "name" (string, max 40 chars) and "amount" (positive number, rounded to 2 decimal places).

Example output: [{"name": "Netflix", "amount": 649.00}, ...]`;

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: { responseMimeType: 'application/json' },
        });

        let predictions;
        try {
            predictions = safeParseAIJson(response.text);
            // Validate structure
            predictions = predictions
                .filter(p => p && typeof p.name === 'string' && typeof p.amount === 'number')
                .slice(0, 3)
                .map(p => ({ name: p.name.slice(0, 40), amount: parseFloat(p.amount.toFixed(2)) }));
        } catch {
            return res.json([]); // Fail gracefully — empty predictions beat an error
        }

        res.json(predictions);
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'Prediction engine failed. Please try again.' });
    }
});

// ─── Health Check ─────────────────────────────────────────────────────────────

app.get('/health', (_req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        auth: adminInitialized ? 'enabled' : 'disabled (dev mode)',
    });
});

// ─── Global Error Handler ─────────────────────────────────────────────────────

app.use((err, _req, res, _next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error.' });
});

// ─── Start ────────────────────────────────────────────────────────────────────


const PORT = process.env.PORT || 3000;

// 🌟 THE FIX: Notice the '0.0.0.0' added right after PORT
app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n🚀 Monix Backend running on port ${PORT}`);
    console.log(`   Health check: http://localhost:${PORT}/health`);
    console.log(`   Auth: ${adminInitialized ? 'Firebase Admin enabled' : 'DEV MODE — auth disabled'}\n`);
});
