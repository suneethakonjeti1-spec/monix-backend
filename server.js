require('dotenv').config(); 
const express = require('express');
const multer = require('multer');
const PDFParser = require("pdf2json"); 
const cors = require('cors');
const { GoogleGenAI } = require('@google/genai');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' })); 

// 🌟 THE FIX: Store the uploaded file directly in RAM (Memory) instead of a folder
const upload = multer({ storage: multer.memoryStorage() });

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY }); 

// 🌟 THE FIX: Parse the PDF directly from the memory buffer
function extractTextFromPDF(pdfBuffer) {
    return new Promise((resolve, reject) => {
        const pdfParser = new PDFParser(null, 1); 
        pdfParser.on("pdfParser_dataError", errData => reject(errData.parserError));
        pdfParser.on("pdfParser_dataReady", () => { resolve(pdfParser.getRawTextContent()); });
        pdfParser.parseBuffer(pdfBuffer); // Read from RAM!
    });
}

// ==========================================
// 🚀 ROUTE 1: UPLOAD & ANALYZE PDF (STATELESS)
// ==========================================
app.post('/upload', upload.single('pdfFile'), async (req, res) => {
    try {
        console.log("\n------------------------------------------------");
        console.log("📥 NEW REQUEST RECEIVED!");
        
        if (!req.file || req.file.mimetype !== 'application/pdf') {
            return res.status(400).json({ error: "Only PDF files are allowed." });
        }
        
        console.log("✅ File is a valid PDF. Extracting from Memory...");
        // 🌟 THE FIX: Pass the buffer directly to our function
        let extractedText = await extractTextFromPDF(req.file.buffer); 

        console.log("🧠 Step 2: Sending to Gemini AI...");
        const prompt = `
        You are a financial AI specialized in Indian Bank Statements and UPI Wallet statements (like PhonePe). 
        Extract all transactions from the following text. 
        CRITICAL RULES:
        1. If you see an "Opening Balance", you can make it the very first transaction.
        2. ONLY extract actual transactions (Money In / Money Out). Look at "Debit", "Credit", or "Amount" columns.
        3. IGNORE running "Balance" columns completely. 
        4. YOU MUST CHOOSE FROM ONLY THESE CATEGORIES: "Utilities", "Salary", "Food & Dining", "Shopping", "Auto & Transport", "Entertainment", "Transfers & Loans", "Groceries", "UPI Payment", or "Other".
        5. NEVER use double quotes (") or backslashes (\\) inside the description string. Strip them out completely.
        Return strictly a JSON array of objects: "date", "description", "amount" (Positive for credit/income, Negative for debit/expense), "category".
        Text: ${extractedText}`; 

        const aiResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
            config: { responseMimeType: "application/json" }
        });

        console.log("🧹 Step 3: Cleaning & Sending back to Frontend...");
        let cleanJSON = aiResponse.text.replace(/```json/g, '').replace(/```/g, '').trim();
        
        if (!cleanJSON.endsWith("]")) {
            let lastValidBracket = cleanJSON.lastIndexOf("}");
            if (lastValidBracket !== -1) { cleanJSON = cleanJSON.substring(0, lastValidBracket + 1) + "]"; }
        }

        res.json(JSON.parse(cleanJSON));

    } catch (error) {
        console.error("Server Error:", error);
        res.status(500).json({ error: "Failed to process statement." });
    }
});

// ==========================================
// 🤖 ROUTE 2: AI DOUBT RESOLVER (CHAT)
// ==========================================
app.post('/chat', async (req, res) => {
    try {
        console.log("💬 Chatbot analyzing context...");
        const { question, history } = req.body; 
        
        const prompt = `You are Monix, a helpful financial assistant. 
        Answer the user's question directly and concisely based ONLY on this transaction data: ${JSON.stringify(history)}. 
        User Question: ${question}`;
        
        const response = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: prompt });
        res.json({ answer: response.text });
    } catch (error) { res.status(500).json({ error: "Failed to connect to AI." }); }
});

// ==========================================
// 🔮 ROUTE 3: NEXT MONTH PREDICTIONS
// ==========================================
app.post('/predict', async (req, res) => {
    try {
        console.log("🔮 Predicting future bills...");
        const { history } = req.body; 
        
        const prompt = `Analyze this transaction history: ${JSON.stringify(history)}. 
        Identify recurring expenses. Predict exactly 3 bills the user will likely have to pay next month.
        Return strictly a JSON array of objects with keys: "name" (string), "amount" (positive number).`;
        
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: { responseMimeType: "application/json" }
        });

        let cleanJSON = response.text.replace(/```json/g, '').replace(/```/g, '').trim();
        res.json(JSON.parse(cleanJSON));
    } catch (error) { res.status(500).json({ error: "Prediction engine failed." }); }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`\n🚀 Secure Stateless Server running on port ${PORT}`);
    console.log(`Waiting for Frontend Requests...\n`);
});