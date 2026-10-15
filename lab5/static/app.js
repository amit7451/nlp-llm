/* =========================================================
   SPORTIX AI — FRONTEND CONTROLLER (JavaScript)
   ========================================================= */

document.addEventListener('DOMContentLoaded', () => {
  const chatContainer = document.getElementById('chatContainer');
  const messagesList = document.getElementById('messagesList');
  const welcomeHero = document.getElementById('welcomeHero');
  const chatForm = document.getElementById('chatForm');
  const chatInput = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');
  const typingIndicator = document.getElementById('typingIndicator');
  const modelSelect = document.getElementById('modelSelect');
  const clearChatBtn = document.getElementById('clearChatBtn');
  const apiKeyBtn = document.getElementById('apiKeyBtn');
  const apiKeyModal = document.getElementById('apiKeyModal');
  const closeModalBtn = document.getElementById('closeModalBtn');
  const customApiKeyInput = document.getElementById('customApiKey');
  const saveApiKeyBtn = document.getElementById('saveApiKeyBtn');

  // Internal conversation memory for context
  let conversationHistory = [];

  // Load custom API key from localStorage if present
  let customApiKey = localStorage.getItem('sportix_custom_api_key') || '';
  if (customApiKey) {
    customApiKeyInput.value = customApiKey;
  }

  // Auto-resize textarea
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
  });

  // Handle enter key submit
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event('submit'));
    }
  });

  // Quick suggestion chips
  document.querySelectorAll('.prompt-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        chatInput.value = prompt;
        chatInput.style.height = 'auto';
        chatForm.dispatchEvent(new Event('submit'));
      }
    });
  });

  // Clear chat
  clearChatBtn.addEventListener('click', () => {
    conversationHistory = [];
    messagesList.innerHTML = '';
    welcomeHero.classList.remove('hidden');
    chatInput.focus();
  });

  // API Key Modal Handlers
  apiKeyBtn.addEventListener('click', () => {
    apiKeyModal.classList.remove('hidden');
    customApiKeyInput.focus();
  });

  closeModalBtn.addEventListener('click', () => {
    apiKeyModal.classList.add('hidden');
  });

  apiKeyModal.addEventListener('click', (e) => {
    if (e.target === apiKeyModal) {
      apiKeyModal.classList.add('hidden');
    }
  });

  saveApiKeyBtn.addEventListener('click', () => {
    customApiKey = customApiKeyInput.value.trim();
    if (customApiKey) {
      localStorage.setItem('sportix_custom_api_key', customApiKey);
    } else {
      localStorage.removeItem('sportix_custom_api_key');
    }
    apiKeyModal.classList.add('hidden');
  });

  // Form Submit Handler
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query) return;

    // Hide welcome hero on first message
    welcomeHero.classList.add('hidden');

    // Add user message to UI
    appendUserMessage(query);
    conversationHistory.push({ role: 'user', content: query });

    // Reset input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatInput.disabled = true;
    sendBtn.disabled = true;

    // Show typing indicator
    showTyping(true);
    scrollToBottom();

    try {
      const payload = {
        messages: conversationHistory,
        model: modelSelect.value,
        api_key: customApiKey || undefined
      };

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error occurred' }));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      const botReply = data.reply;

      conversationHistory.push({ role: 'assistant', content: botReply });
      appendBotMessage(botReply);
    } catch (err) {
      console.error('Chat error:', err);
      appendErrorMessage(err.message);
    } finally {
      showTyping(false);
      chatInput.disabled = false;
      sendBtn.disabled = false;
      chatInput.focus();
      scrollToBottom();
    }
  });

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar user-avatar';
    avatar.textContent = 'YOU';

    const contentWrapper = document.createElement('div');
    contentWrapper.className = 'message-content-wrapper';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble user-bubble';
    bubble.textContent = text;

    contentWrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(contentWrapper);
    messagesList.appendChild(row);
  }

  function appendBotMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row bot-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"></path>
        <path d="M2 12h20"></path>
      </svg>
    `;

    const contentWrapper = document.createElement('div');
    contentWrapper.className = 'message-content-wrapper';

    // Detect if the bot triggered the sports guardrail refusal
    const isRejection = isGuardrailRejection(text);
    if (isRejection) {
      const notice = document.createElement('div');
      notice.className = 'rejection-notice';
      notice.innerHTML = `<span>🛡️ Domain Guardrail Triggered: Non-Sports Query</span>`;
      contentWrapper.appendChild(notice);
    }

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-bubble';
    bubble.innerHTML = formatMarkdown(text);

    contentWrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(contentWrapper);
    messagesList.appendChild(row);
  }

  function appendErrorMessage(errorMsg) {
    const row = document.createElement('div');
    row.className = 'message-row bot-row';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar bot-avatar';
    avatar.style.borderColor = 'var(--accent-danger)';
    avatar.style.color = 'var(--accent-danger)';
    avatar.textContent = '!';

    const contentWrapper = document.createElement('div');
    contentWrapper.className = 'message-content-wrapper';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble bot-bubble';
    bubble.style.borderColor = 'rgba(239, 68, 68, 0.4)';
    bubble.innerHTML = `<span style="color: #fca5a5; font-weight: 600;">Error:</span> ${escapeHtml(errorMsg)}`;

    contentWrapper.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(contentWrapper);
    messagesList.appendChild(row);
  }

  function isGuardrailRejection(text) {
    const lower = text.toLowerCase();
    return (
      lower.includes('only assist with sports') ||
      lower.includes('only answer sports') ||
      lower.includes('dedicated exclusively to the sports domain') ||
      lower.includes('abstain from answering') ||
      lower.includes('sports-related questions') && (lower.includes('cannot answer') || lower.includes('cannot assist') || lower.includes('only answer'))
    );
  }

  function showTyping(show) {
    if (show) {
      typingIndicator.classList.remove('hidden');
    } else {
      typingIndicator.classList.add('hidden');
    }
  }

  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Lightweight markdown formatter for bolding, bullet points, numbered lists, and linebreaks
  function formatMarkdown(text) {
    let sanitized = escapeHtml(text);

    // Bold **text**
    sanitized = sanitized.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Italic *text*
    sanitized = sanitized.replace(/\*(.*?)\*/g, '<em>$1</em>');

    // Inline code `code`
    sanitized = sanitized.replace(/`(.*?)`/g, '<code style="background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 4px; font-family: monospace;">$1</code>');

    // Bullet points
    sanitized = sanitized.replace(/^[\*\-]\s+(.*)$/gm, '<li style="margin-left: 20px;">$1</li>');

    // Line breaks
    sanitized = sanitized.replace(/\n/g, '<br>');

    return sanitized;
  }
});
