require('dotenv').config();
const express    = require('express');
const multer     = require('multer');
const PDFParser  = require('pdf2json');
const cors       = require('cors');
const morgan     = require('morgan');
const rateLimit  = require('express-rate-limit');
const { GoogleGenAI } = require('@google/genai');
const admin      = require('firebase-admin');

// ─── App ──────────────────────────────────────────────────────
const app = express();
app.set('trust proxy', 1);
app.use(morgan('tiny')); // request logging

// CORS — restrict to your actual frontend origin in production
const ALLOWED_ORIGINS = (process.env.ALLOWED_ORIGINS || '*').split(',').map(s => s.trim());
app.use(cors({
    origin: (origin, cb) => {
        if (ALLOWED_ORIGINS.includes('*') || !origin || ALLOWED_ORIGINS.includes(origin)) return cb(null, true);
        cb(new Error(`CORS: origin ${origin} not allowed`));
    },
    methods: ['GET', 'POST'],
}));

app.use(express.json({ limit: '1mb' }));

// ─── Rate Limiting ─────────────────────────────────────────────
const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, 
    max: 60,                   
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Too many requests. Please wait a few minutes and try again.' },
});

const uploadLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, 
    max: 10,                   
    message: { error: 'Upload limit reached. You can upload up to 10 statements per hour.' },
});

const chatLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 20,
    message: { error: 'Slow down! Max 20 messages per minute.' },
});

const predictLimiter = rateLimit({
    windowMs: 60 * 1000,
    max: 10,
    message: { error: 'Too many prediction requests.' },
});

app.use(apiLimiter); // applies globally to all routes

// ─── Firebase Admin ────────────────────────────────────────────
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
    console.warn('⚠️  Firebase Admin SDK not initialized — auth middleware disabled (dev mode).');
    console.warn('   Set FIREBASE_SERVICE_ACCOUNT env var to enable auth.');
}

// ─── Multer ────────────────────────────────────────────────────
const MAX_MB = 15;
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: MAX_MB * 1024 * 1024 },
    fileFilter: (_req, file, cb) => {
        if (file.mimetype !== 'application/pdf') return cb(new Error('Only PDF files are accepted.'), false);
        cb(null, true);
    },
});

// ─── API KEY ROTATION ENGINE ───────────────────────────────────
// Grabs the 3 keys from Render. If any are missing, it filters them out.
const API_KEYS = [
    process.env.GEMINI_KEY_1,
    process.env.GEMINI_KEY_2,
    process.env.GEMINI_KEY_3,
    process.env.GEMINI_API_KEY // Fallback to your original key just in case
].filter(key => key && key.trim() !== '');

let currentKeyIndex = 0;

function getActiveKey() {
    if (API_KEYS.length === 0) {
        console.error("🚨 CRITICAL ERROR: NO API KEYS FOUND IN ENVIRONMENT VARIABLES.");
        return null;
    }
    return API_KEYS[currentKeyIndex];
}

function rotateKey() {
    if (API_KEYS.length <= 1) return; // Can't rotate if we only have 1 key
    currentKeyIndex++;
    if (currentKeyIndex >= API_KEYS.length) {
        currentKeyIndex = 0; // Loop back to the beginning
    }
    console.log(`🔄 AI Limit hit! Switched to API Key #${currentKeyIndex + 1}`);
}
// ───────────────────────────────────────────────────────────────

// ─── Auth Middleware ───────────────────────────────────────────
async function requireAuth(req, res, next) {
    if (!adminInitialized) return next(); // dev mode bypass

    const header = req.headers.authorization;
    if (!header?.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'Unauthorized: missing Bearer token.' });
    }
    try {
        req.user = await admin.auth().verifyIdToken(header.split('Bearer ')[1]);
        next();
    } catch {
        return res.status(403).json({ error: 'Forbidden: invalid or expired token.' });
    }
}

// ─── Helpers ───────────────────────────────────────────────────
function extractTextFromPDF(buffer) {
    return new Promise((resolve, reject) => {
        const parser = new PDFParser(null, 1);
        parser.on('pdfParser_dataError', e => reject(new Error(e.parserError)));
        parser.on('pdfParser_dataReady', () => resolve(parser.getRawTextContent()));
        parser.parseBuffer(buffer);
    });
}

function safeParseJSON(raw) {
    let s = raw.replace(/```json/gi, '').replace(/```/g, '').trim();
    if (s.startsWith('[') && !s.endsWith(']')) {
        const last = s.lastIndexOf('}');
        s = last !== -1 ? s.slice(0, last + 1) + ']' : s + ']';
    }
    return JSON.parse(s);
}

// ─── ROUTE: Health check ───────────────────────────────────────
app.get('/health', (_req, res) => {
    res.json({
        status: 'ok',
        timestamp: new Date().toISOString(),
        auth: adminInitialized ? 'firebase-admin' : 'dev-bypass',
        version: '2.1.0',
        activeKeysCount: API_KEYS.length
    });
});

// ─── ROUTE: Upload PDF ─────────────────────────────────────────
app.post('/upload', uploadLimiter, requireAuth, upload.single('pdfFile'), async (req, res) => {
    try {
        const uid = req.user?.uid ?? 'dev';
        console.log(`📥 /upload | uid:${uid} | file:${req.file?.originalname}`);

        if (!req.file) return res.status(400).json({ error: 'No PDF file received.' });

        let text;
        try {
            text = await extractTextFromPDF(req.file.buffer);
        } catch {
            return res.status(422).json({
                error: 'Could not read this PDF. Only text-based PDFs are supported — scanned images won\'t work.',
            });
        }

        if (!text || text.trim().length < 50) {
            return res.status(422).json({ error: 'PDF appears empty or has no readable text.' });
        }

        const prompt = `You are an expert financial AI specializing in Indian bank statements.
Extract ALL transactions from the text. Follow these rules exactly:

CRITICAL RULES FOR DEBITS vs CREDITS:
The text columns are mixed up. You MUST use the running balance (the last number in the row) to determine if the transaction amount is a Debit (-) or Credit (+).
1. If the balance DECREASES from the previous row = DEBIT (return as a NEGATIVE number).
2. If the balance INCREASES from the previous row = CREDIT (return as a POSITIVE number).

OPENING BALANCE INSTRUCTION:
3. You MUST include the "OPENING BALANCE" as the very first transaction. Treat it as a POSITIVE number (Credit) and categorize it as "Other".
4. Ignore "CLOSING BALANCE" and "TRANSACTION TOTAL" rows.

CONTEXT CLUES (Axis Bank):
- "UPI/P2M", "UPI/P2A", "Car ch", "Tifin" are almost always DEBITS (-)
- "ACH-CR", "Int.Pd", and "RTGS/NEFT...LOAN RETURN" are CREDITS (+)

OTHER RULES:
5. Assign exactly one category from: "Utilities" | "Salary" | "Food & Dining" | "Shopping" | "Auto & Transport" | "Entertainment" | "Transfers & Loans" | "Groceries" | "UPI Payment" | "Other"
6. Date format: YYYY-MM-DD.
7. description: plain text only, no quotes, max 120 chars.
8. Return ONLY a valid JSON array. No markdown. Each item: { "date", "description", "amount", "category" }

Statement:
${text.slice(0, 15000)}`;

        let aiText;
        try {
            // INITIALIZE AI WITH THE ACTIVE KEY
            const ai = new GoogleGenAI({ apiKey: getActiveKey() });
            
            const r = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: [{ role: 'user', parts: [{ text: prompt }] }],
                config: { responseMimeType: 'application/json' },
            });
            aiText = r.text;
        } catch (e) {
            console.error('Gemini error:', e.message);
            // CHECK FOR RATE LIMITS AND ROTATE
            if (e.message && (e.message.includes('429') || e.message.includes('503') || e.message.includes('RESOURCE_EXHAUSTED'))) {
                rotateKey();
                return res.status(502).json({ error: 'AI servers busy. Key rotated! Please click upload again.' });
            }
            return res.status(502).json({ error: 'AI service unavailable. Please try again shortly.' });
        }

        let parsed;
        try {
            parsed = safeParseJSON(aiText);
        } catch {
            console.error('JSON parse failed. Raw:', aiText?.slice(0, 300));
            return res.status(422).json({ error: 'AI returned malformed data. Please retry — complex PDFs sometimes need a second attempt.' });
        }

        const clean = parsed
            .filter(t => t && typeof t.amount === 'number' && !isNaN(t.amount) && t.date && t.description)
            .map(t => ({
                date: String(t.date).trim().slice(0, 10),
                description: String(t.description).replace(/[<>"'\\]/g, '').trim().slice(0, 120),
                amount: parseFloat(t.amount.toFixed(2)),
                category: t.category || 'Other',
            }));

        console.log(`✅ Extracted ${clean.length} transactions for uid:${uid}`);
        res.json(clean);

    } catch (err) {
        console.error('Unexpected /upload error:', err);
        res.status(500).json({ error: 'Unexpected server error. Please try again.' });
    }
});

// Multer error handler
app.use((err, req, res, next) => {
    if (err.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: `File too large (max ${MAX_MB} MB).` });
    if (err.message === 'Only PDF files are accepted.') return res.status(400).json({ error: err.message });
    next(err);
});

// ─── ROUTE: Chat ───────────────────────────────────────────────
app.post('/chat', chatLimiter, requireAuth, async (req, res) => {
    try {
        const { question, history = [], roastMode = false } = req.body;

        if (!question?.trim()) return res.status(400).json({ error: 'Question cannot be empty.' });
        if (question.length > 800) return res.status(400).json({ error: 'Question too long (max 800 chars).' });

        const spend  = history.filter(t => t.amount < 0).reduce((s, t) => s + Math.abs(t.amount), 0);
        const income = history.filter(t => t.amount > 0).reduce((s, t) => s + t.amount, 0);

        const persona = roastMode
            ? `You are Monix in Roast Mode — brutally honest, sarcastic, slightly savage. Call out bad spending habits with dark humor. Keep it under 3 sentences. End with one actionable tip.`
            : `You are Monix, a concise professional Indian personal finance advisor. Be specific and reference actual data. Use ₹. Under 120 words.`;

        const prompt = `${persona}

Context:
- Income: ₹${income.toFixed(0)}  Spending: ₹${spend.toFixed(0)}  Net: ₹${(income-spend).toFixed(0)}
- Transactions: ${history.length} total
- Recent 15: ${JSON.stringify(history.slice(-15))}

Question: ${question}`;

        try {
            // INITIALIZE AI WITH THE ACTIVE KEY
            const ai = new GoogleGenAI({ apiKey: getActiveKey() });
            const r = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: prompt });
            res.json({ answer: r.text });
        } catch (e) {
            console.error('Gemini chat error:', e.message);
            if (e.message && (e.message.includes('429') || e.message.includes('503') || e.message.includes('RESOURCE_EXHAUSTED'))) {
                rotateKey();
                return res.status(502).json({ error: 'AI busy. Key rotated! Try asking again.' });
            }
            throw e; // pass to outer catch
        }

    } catch (err) {
        console.error('/chat error:', err);
        res.status(500).json({ error: 'AI unavailable. Please try again.' });
    }
});

// ─── ROUTE: Predict ────────────────────────────────────────────
app.post('/predict', predictLimiter, requireAuth, async (req, res) => {
    try {
        const { history = [] } = req.body;
        if (history.length < 5) return res.json([]);

        const prompt = `Analyze this Indian bank transaction history and find recurring monthly expenses.

Transactions: ${JSON.stringify(history)}

Rules:
1. Only include bills that appear at least twice with similar amounts (±20% variance OK).
2. Return exactly 3 predictions. If fewer than 3 recurring bills exist, return fewer.
3. Use real names (e.g. "Netflix Subscription", "Electricity Bill", "Gym Membership").
4. Return ONLY a valid JSON array, no markdown. Each item: { "name": string, "amount": positive number }`;

        let r;
        try {
            // INITIALIZE AI WITH THE ACTIVE KEY
            const ai = new GoogleGenAI({ apiKey: getActiveKey() });
            r = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: prompt,
                config: { responseMimeType: 'application/json' },
            });
        } catch (e) {
             console.error('Gemini predict error:', e.message);
             if (e.message && (e.message.includes('429') || e.message.includes('503') || e.message.includes('RESOURCE_EXHAUSTED'))) {
                 rotateKey();
                 // We fail silently here so the frontend just tries again later without breaking the UI
                 return res.json([]); 
             }
             throw e;
        }

        let preds;
        try {
            preds = safeParseJSON(r.text);
            preds = preds
                .filter(p => p?.name && typeof p.amount === 'number' && p.amount > 0)
                .slice(0, 3)
                .map(p => ({ name: String(p.name).slice(0, 50), amount: parseFloat(p.amount.toFixed(2)) }));
        } catch { return res.json([]); }

        res.json(preds);
    } catch (err) {
        console.error('/predict error:', err);
        res.status(500).json({ error: 'Prediction failed.' });
    }
});

// ─── Global error handler ──────────────────────────────────────
app.use((err, _req, res, _next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error.' });
});

// ─── Start ─────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n🚀 Monix Backend v2.1 | port ${PORT}`);
    console.log(`   Health : http://localhost:${PORT}/health`);
    console.log(`   Auth   : ${adminInitialized ? 'Firebase Admin ✓' : 'DEV bypass'}`);
    console.log(`   CORS   : ${ALLOWED_ORIGINS.join(', ')}\n`);
});