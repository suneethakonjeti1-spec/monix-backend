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

// ─── API KEYS ──────────────────────────────────────────────────
const API_KEYS = [
    process.env.GEMINI_KEY_1,
    process.env.GEMINI_KEY_2,
    process.env.GEMINI_KEY_3,
    process.env.GEMINI_API_KEY,
].filter(k => k && k.trim() !== '');

if (API_KEYS.length === 0) console.error('🚨 CRITICAL: NO GEMINI API KEYS FOUND.');
else console.log(`✅ Loaded ${API_KEYS.length} Gemini key(s).`);

// ─── RESILIENT AI ENGINE ───────────────────────────────────────
const sleep = ms => new Promise(r => setTimeout(r, ms));
const BACKOFF_MS    = [2000, 5000, 10000];
const GEMINI_MODELS = ['gemini-2.5-flash', 'gemini-2.0-flash'];

const IS_OVERLOAD = msg => msg && (
    msg.includes('429') || msg.includes('503') ||
    msg.includes('RESOURCE_EXHAUSTED') || msg.includes('UNAVAILABLE')
);
const IS_MODEL_ERROR = msg => msg && (
    msg.includes('404') || msg.includes('NOT_FOUND') || msg.includes('not found')
);

async function tryGemini(apiKey, model, prompt, config = {}) {
    const ai = new GoogleGenAI({ apiKey });
    const r  = await ai.models.generateContent({
        model,
        contents: typeof prompt === 'string'
            ? [{ role: 'user', parts: [{ text: prompt }] }]
            : prompt,
        config,
    });
    return r.text;
}

// ✅ FIXED GROQ FUNCTION
async function tryGroq(prompt) {
    const key = process.env.GROQ_API_KEY;
    if (!key) throw new Error('No GROQ_API_KEY set.');
    
    let text = typeof prompt === 'string' ? prompt
        : prompt?.[0]?.parts?.[0]?.text ?? JSON.stringify(prompt);

    // Slice text to prevent 400 errors from massive files
    if (text.length > 8000) {
        text = text.slice(0, 8000);
    }

    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method : 'POST',
        headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
        body   : JSON.stringify({
            model      : 'llama-3.1-8b-instant', // Correct, fast model
            messages   : [
                { role: 'system', content: 'You are a data extractor. Return ONLY a valid JSON array of transaction objects. No explanations or markdown wrappers.' },
                { role: 'user', content: text }
            ],
            temperature: 0.1,
        }),
    });
    if (!res.ok) throw new Error(`Groq ${res.status}`);
    const data = await res.json();
    return data.choices[0].message.content;
}

async function callAI(prompt, config = {}) {
    let attempt = 0;

    for (const model of GEMINI_MODELS) {
        for (let i = 0; i < API_KEYS.length; i++) {
            attempt++;
            try {
                console.log(`🤖 Attempt ${attempt}: ${model} key #${i + 1}`);
                const text = await tryGemini(API_KEYS[i], model, prompt, config);
                console.log(`✅ Success on attempt ${attempt}`);
                return text;
            } catch (e) {
                console.error(`❌ Attempt ${attempt} failed: ${e.message?.slice(0, 120)}`);
                if (IS_MODEL_ERROR(e.message)) {
                    console.log(`⚠️  Model ${model} not available in this API version — skipping model.`);
                    break;
                }
                if (!IS_OVERLOAD(e.message)) throw e;
                const wait = BACKOFF_MS[i] ?? 10000;
                console.log(`⏳ Overload detected — waiting ${wait / 1000}s…`);
                await sleep(wait);
            }
        }
        console.log(`🔄 All keys exhausted for ${model}, trying next model…`);
    }

    try {
        attempt++;
        console.log(`🤖 Attempt ${attempt}: Groq llama-3.1-8b-instant (final fallback)`);
        const text = await tryGroq(prompt);
        console.log('✅ Groq fallback succeeded.');
        return text;
    } catch (e) {
        console.error(`❌ Groq failed: ${e.message}`);
    }

    throw new Error('All AI services are currently overloaded. Please try again in a few minutes.');
}
// ───────────────────────────────────────────────────────────────

// ─── Auth Middleware ───────────────────────────────────────────
async function requireAuth(req, res, next) {
    if (!adminInitialized) return next(); 

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
        version: '2.2.0',
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
                error: 'Could not read this PDF. Only text-based PDFs are supported.',
            });
        }

        if (!text || text.trim().length < 50) {
            return res.status(422).json({ error: 'PDF appears empty or has no readable text.' });
        }

        // ✅ STRONGER, MORE RELIABLE PROMPT
        const prompt = `You are a precision financial data extraction AI. Process this Indian bank statement text and extract all transactions into a structured JSON array.

CRITICAL LOGIC FOR DEBIT VS CREDIT:
Bank statement columns are often misaligned. You MUST use the RUNNING BALANCE (the last number in a row) to determine the sign:
1. If the running balance DECREASES from the previous transaction, it is a DEBIT. Return the amount as a NEGATIVE number (e.g., -500).
2. If the running balance INCREASES from the previous transaction, it is a CREDIT. Return the amount as a POSITIVE number (e.g., 500).

INCLUSION RULES:
3. OPENING BALANCE: If present, include it as the first transaction. Amount is POSITIVE. Category: "Other".
4. IGNORE: "Closing Balance", "Transaction Total", "B/F", "C/F", or header rows. Include ONLY actual transactions.

CATEGORIZATION RULES:
5. Assign EXACTLY ONE category from this list: "Utilities", "Salary", "Food & Dining", "Shopping", "Auto & Transport", "Entertainment", "Transfers & Loans", "Groceries", "UPI Payment", "Other".
6. Context Clues: "UPI/P2M", "Zomato", "Swiggy" -> Food/Groceries. "ACH-CR", "SALARY" -> Salary. "EMI", "LOAN" -> Transfers & Loans.

FORMATTING RULES:
7. Date: YYYY-MM-DD format only.
8. Description: Clean, plain text. Remove excess spaces or asterisks. Max 100 characters.
9. Amount: Float/Number type, no currency symbols or commas.
10. OUTPUT: Return ONLY a valid JSON array of objects. No markdown.
Example: [{"date": "2024-03-01", "description": "OPENING BALANCE", "amount": 15000.00, "category": "Other"}]

Statement Data:
${text.slice(0, 15000)}`;

        let aiText;
        try {
            aiText = await callAI(
                [{ role: 'user', parts: [{ text: prompt }] }],
                { responseMimeType: 'application/json' }
            );
        } catch (e) {
            console.error('/upload AI error:', e.message);
            return res.status(503).json({ error: e.message });
        }

        let parsed;
        try {
            parsed = safeParseJSON(aiText);
        } catch {
            console.error('JSON parse failed. Raw:', aiText?.slice(0, 300));
            return res.status(422).json({ error: 'AI returned malformed data. Please retry.' });
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

        let answer;
        try {
            const text = await callAI(prompt);
            answer = text;
        } catch (e) {
            console.error('/chat AI error:', e.message);
            return res.status(503).json({ error: e.message });
        }
        res.json({ answer });

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

        let rawText;
        try {
            rawText = await callAI(prompt, { responseMimeType: 'application/json' });
        } catch (e) {
            console.error('/predict AI error:', e.message);
            return res.json([]); 
        }

        let preds;
        try {
            preds = safeParseJSON(rawText);
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

app.use((err, _req, res, _next) => {
    console.error('Unhandled error:', err);
    res.status(500).json({ error: 'Internal server error.' });
});

// ─── Start ─────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', () => {
    console.log(`\n🚀 Monix Backend v2.2 | port ${PORT}`);
    console.log(`   Health : http://localhost:${PORT}/health`);
    console.log(`   Auth   : ${adminInitialized ? 'Firebase Admin ✓' : 'DEV bypass'}`);
    console.log(`   CORS   : ${ALLOWED_ORIGINS.join(', ')}`);
    console.log(`   AI     : ${API_KEYS.length} Gemini key(s) + ${process.env.GROQ_API_KEY ? 'Groq ✓' : 'Groq ✗ (add GROQ_API_KEY)'}\n`);
});