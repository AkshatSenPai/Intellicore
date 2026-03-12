/**
 * INTELLICORE - Chat Interface (API Connected)
 * Connects to the Flask backend for RAG-powered persona responses.
 * Supports both streaming and non-streaming modes.
 */

// ── Configuration ──────────────────────────────
const API_BASE = "http://localhost:5000";
const USE_STREAMING = true; // Set to false to use non-streaming mode

// ── Chat State ─────────────────────────────────
let currentCredits = 50;
const maxCredits = 50;
let conversationHistory = [];
let isWaitingForResponse = false;

// ── Initialize ─────────────────────────────────
document.addEventListener("DOMContentLoaded", function () {
    initializeChat();
    checkServerStatus();
});

function initializeChat() {
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendButton");
    const messagesArea = document.getElementById("messagesArea");

    if (!messageInput || !sendButton) return;

    // Scroll to bottom initially
    if (messagesArea) {
        setTimeout(() => {
            messagesArea.scrollTop = messagesArea.scrollHeight;
        }, 100);
    }

    // Send on button click
    sendButton.addEventListener("click", sendMessage);

    // Send on Enter
    messageInput.addEventListener("keypress", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    updateCreditDisplay();
}

/**
 * Check if the backend server is running
 */
async function checkServerStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        const data = await res.json();

        if (data.status === "ok") {
            console.log("✅ Server connected:", data);
            if (!data.model_available) {
                showSystemMessage(
                    `⚠️ Model "${data.model}" not found in Ollama. Run: ollama pull ${data.model}`
                );
            }
            if (data.db_chunks === 0) {
                showSystemMessage(
                    "⚠️ Vector database is empty. Run: python build_chroma.py"
                );
            }
        }
    } catch (err) {
        console.error("❌ Server not reachable:", err);
        showSystemMessage(
            "❌ Cannot connect to server. Make sure to run: python server.py"
        );
    }
}

/**
 * Show a system/status message in the chat
 */
function showSystemMessage(text) {
    const messagesArea = document.getElementById("messagesArea");
    if (!messagesArea) return;

    const div = document.createElement("div");
    div.className = "message system-message";
    div.innerHTML = `
        <div class="message-content" style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); max-width: 80%; margin: 0 auto; text-align: center;">
            <p style="color: #f59e0b; font-size: 0.875rem;">${text}</p>
        </div>
    `;
    messagesArea.appendChild(div);
    scrollToBottom(messagesArea);
}

// ── Send Message ───────────────────────────────

async function sendMessage() {
    const messageInput = document.getElementById("messageInput");
    const text = messageInput.value.trim();

    if (!text || currentCredits <= 0 || isWaitingForResponse) return;

    // Add user message to UI
    addMessage(text, "user");
    messageInput.value = "";

    // Decrease credits
    currentCredits--;
    updateCreditDisplay();

    // Lock input while waiting
    isWaitingForResponse = true;
    setInputState(false);

    if (USE_STREAMING) {
        await sendStreamingMessage(text);
    } else {
        await sendNonStreamingMessage(text);
    }

    // Unlock input
    isWaitingForResponse = false;
    if (currentCredits > 0) {
        setInputState(true);
        messageInput.focus();
    }

    // Check credits
    if (currentCredits <= 0) {
        handleCreditsExhausted();
    }
}

/**
 * Non-streaming: send message and wait for full response
 */
async function sendNonStreamingMessage(text) {
    // Show typing indicator
    const typingId = showTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });

        removeTypingIndicator(typingId);

        if (!res.ok) {
            const err = await res.json();
            addMessage(`Error: ${err.error || "Server error"}`, "system");
            return;
        }

        const data = await res.json();
        addMessage(data.response, "persona");

        if (data.sources && data.sources.length > 0) {
            console.log("📂 Sources:", data.sources);
        }
    } catch (err) {
        removeTypingIndicator(typingId);
        addMessage("❌ Failed to connect to server. Is it running?", "system");
        console.error("Chat error:", err);
    }
}

/**
 * Streaming: send message and display tokens as they arrive
 */
async function sendStreamingMessage(text) {
    // Create an empty persona message bubble to fill
    const { messageEl, contentEl } = createEmptyPersonaMessage();

    try {
        const res = await fetch(`${API_BASE}/api/chat/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text }),
        });

        if (!res.ok) {
            const err = await res.json();
            contentEl.querySelector("p").textContent = `Error: ${err.error || "Server error"}`;
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = "";
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE lines
            const lines = buffer.split("\n");
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;

                try {
                    const data = JSON.parse(line.slice(6));

                    if (data.token) {
                        fullResponse += data.token;
                        contentEl.querySelector("p").textContent = fullResponse;
                        const messagesArea = document.getElementById("messagesArea");
                        scrollToBottom(messagesArea);
                    }

                    if (data.done) {
                        console.log("📂 Sources:", data.sources);
                    }

                    if (data.error) {
                        contentEl.querySelector("p").textContent = `Error: ${data.error}`;
                    }
                } catch (e) {
                    // Skip malformed JSON
                }
            }
        }

        // Store in history
        if (fullResponse) {
            conversationHistory.push({
                type: "user",
                text: text,
                timestamp: formatTimestamp(),
            });
            conversationHistory.push({
                type: "persona",
                text: fullResponse,
                timestamp: formatTimestamp(),
            });
        }
    } catch (err) {
        contentEl.querySelector("p").textContent =
            "❌ Failed to connect to server. Is it running?";
        console.error("Stream error:", err);
    }
}

// ── UI Helpers ─────────────────────────────────

/**
 * Add a complete message to the chat
 */
function addMessage(text, type) {
    const messagesArea = document.getElementById("messagesArea");
    if (!messagesArea) return;

    // Remove welcome message on first real message
    const welcome = messagesArea.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const timestamp = formatTimestamp();
    const messageDiv = document.createElement("div");
    messageDiv.className = `message ${type}-message`;

    if (type === "user") {
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${escapeHtml(text)}</p>
            </div>
            <div class="message-meta">
                <span class="message-sender">You</span>
                <span class="message-time">${timestamp}</span>
            </div>
        `;
    } else if (type === "persona") {
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <div class="persona-avatar small trump-avatar">DT</div>
            </div>
            <div class="message-wrapper">
                <div class="message-content">
                    <p>${escapeHtml(text)}</p>
                </div>
                <div class="message-meta">
                    <span class="message-sender">Donald Trump</span>
                    <span class="message-time">${timestamp}</span>
                </div>
            </div>
        `;
    } else {
        // System message
        messageDiv.innerHTML = `
            <div class="message-content" style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); max-width: 80%; margin: 0 auto; text-align: center;">
                <p style="color: #f59e0b; font-size: 0.875rem;">${text}</p>
            </div>
        `;
    }

    messagesArea.appendChild(messageDiv);
    scrollToBottom(messagesArea);

    // Store in history (non-streaming)
    if (type !== "system" && !USE_STREAMING) {
        conversationHistory.push({ type, text, timestamp });
    }
}

/**
 * Create an empty persona message bubble for streaming
 */
function createEmptyPersonaMessage() {
    const messagesArea = document.getElementById("messagesArea");

    // Remove welcome message
    const welcome = messagesArea.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const timestamp = formatTimestamp();
    const messageDiv = document.createElement("div");
    messageDiv.className = "message persona-message";
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <div class="persona-avatar small trump-avatar">DT</div>
        </div>
        <div class="message-wrapper">
            <div class="message-content">
                <p><span class="typing-dots">●●●</span></p>
            </div>
            <div class="message-meta">
                <span class="message-sender">Donald Trump</span>
                <span class="message-time">${timestamp}</span>
            </div>
        </div>
    `;

    messagesArea.appendChild(messageDiv);
    scrollToBottom(messagesArea);

    return {
        messageEl: messageDiv,
        contentEl: messageDiv.querySelector(".message-content"),
    };
}

/**
 * Show typing indicator
 */
function showTypingIndicator() {
    const messagesArea = document.getElementById("messagesArea");
    const id = "typing-" + Date.now();

    const div = document.createElement("div");
    div.className = "message persona-message";
    div.id = id;
    div.innerHTML = `
        <div class="message-avatar">
            <div class="persona-avatar small trump-avatar">DT</div>
        </div>
        <div class="message-wrapper">
            <div class="message-content">
                <p class="typing-indicator">
                    <span class="typing-dots">●●●</span>
                </p>
            </div>
        </div>
    `;

    messagesArea.appendChild(div);
    scrollToBottom(messagesArea);
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

/**
 * Enable/disable input area
 */
function setInputState(enabled) {
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendButton");

    if (messageInput) {
        messageInput.disabled = !enabled;
        if (enabled) {
            messageInput.placeholder = "Type your message...";
        } else if (currentCredits > 0) {
            messageInput.placeholder = "Waiting for response...";
        }
    }
    if (sendButton) {
        sendButton.disabled = !enabled;
    }
}

/**
 * Update credit display
 */
function updateCreditDisplay() {
    const creditCount = document.getElementById("creditCount");
    const remainingCredits = document.getElementById("remainingCredits");
    const creditCounter = document.getElementById("creditCounter");
    const messageInput = document.getElementById("messageInput");
    const sendButton = document.getElementById("sendButton");
    const inputStatus = document.getElementById("inputStatus");

    if (creditCount) creditCount.textContent = currentCredits;
    if (remainingCredits) remainingCredits.textContent = currentCredits;

    if (creditCounter) {
        if (currentCredits === 0) {
            creditCounter.style.backgroundColor = "rgba(220, 38, 38, 0.2)";
        } else if (currentCredits < 10) {
            creditCounter.style.backgroundColor = "rgba(245, 158, 11, 0.2)";
        }
    }

    if (currentCredits <= 0) {
        if (messageInput) {
            messageInput.disabled = true;
            messageInput.placeholder = "No credits remaining";
        }
        if (sendButton) sendButton.disabled = true;
        if (inputStatus) {
            inputStatus.innerHTML = "🔒 No credits remaining";
            inputStatus.style.color = "var(--color-locked)";
        }
    }
}

/**
 * Handle credits exhausted
 */
function handleCreditsExhausted() {
    setTimeout(() => {
        window.IntellicoreApp.showModal("creditsModal");
    }, 1500);
}

// ── Utility Functions ──────────────────────────

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function formatTimestamp(date = new Date()) {
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const ampm = hours >= 12 ? "PM" : "AM";
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes.toString().padStart(2, "0");
    return `${displayHours}:${displayMinutes} ${ampm}`;
}

function scrollToBottom(element) {
    if (element) {
        element.scrollTop = element.scrollHeight;
    }
}

// ── Exports ────────────────────────────────────

window.ChatInterface = {
    addMessage,
    getConversationHistory: () => conversationHistory,
    currentCredits: () => currentCredits,
    exportConversation: function () {
        let text = "Conversation with Donald Trump\n";
        text += "================================\n\n";
        conversationHistory.forEach((msg) => {
            const sender = msg.type === "user" ? "You" : "Donald Trump";
            text += `[${msg.timestamp}] ${sender}:\n${msg.text}\n\n`;
        });
        return text;
    },
};
