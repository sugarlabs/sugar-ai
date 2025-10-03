document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const chatSubmit = document.getElementById('chat-submit');
    const useRag = document.getElementById('use-rag');
    const quotaRemaining = document.getElementById('quota-remaining');
    const quotaTotal = document.getElementById('quota-total');
    const quotaBar = document.querySelector('.quota-bar');
    const apiKeyField = document.getElementById('api-key-field');
    const toggleApiKeyBtn = document.getElementById('toggle-api-key');
    const copyApiKeyBtn = document.getElementById('copy-api-key');
    
    // New elements for endpoint selection
    const endpointRadios = document.querySelectorAll('input[name="endpoint-choice"]');
    const customPromptOptions = document.getElementById('custom-prompt-options');
    const customPromptTextarea = document.getElementById('custom-prompt');
    const presetBtns = document.querySelectorAll('.preset-btn');
    
    // Parameter elements
    const temperatureInput = document.getElementById('temperature');
    const topPInput = document.getElementById('top-p');
    const topKInput = document.getElementById('top-k');
    const maxLengthInput = document.getElementById('max-length');
    const repetitionPenaltyInput = document.getElementById('repetition-penalty');
    const truncationInput = document.getElementById('truncation');
    
    // add a user message to the chat
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.style.textAlign = 'right';
        messageElement.style.marginBottom = '10px';
        
        const textElement = document.createElement('span');
        textElement.style.background = '#3498db';
        textElement.style.color = 'white';
        textElement.style.padding = '8px 12px';
        textElement.style.borderRadius = '15px 15px 0 15px';
        textElement.style.display = 'inline-block';
        textElement.style.maxWidth = '80%';
        textElement.textContent = message;
        
        messageElement.appendChild(textElement);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // add an AI message to the chat
    function addAIMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.style.textAlign = 'left';
        messageElement.style.marginBottom = '10px';
        
        const textElement = document.createElement('span');
        textElement.style.background = '#f0f0f0';
        textElement.style.padding = '8px 12px';
        textElement.style.borderRadius = '15px 15px 15px 0';
        textElement.style.display = 'inline-block';
        textElement.style.maxWidth = '80%';
        
        // convert markdown-like syntax to HTML
        let formattedMessage = message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        textElement.innerHTML = formattedMessage;
        
        messageElement.appendChild(textElement);
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // add a "typing" indicator
    function showTypingIndicator() {
        const indicatorElement = document.createElement('div');
        indicatorElement.id = 'typing-indicator';
        indicatorElement.style.textAlign = 'left';
        indicatorElement.style.marginBottom = '10px';
        
        const textElement = document.createElement('span');
        textElement.style.background = '#f0f0f0';
        textElement.style.padding = '8px 12px';
        textElement.style.borderRadius = '15px';
        textElement.style.display = 'inline-block';
        textElement.innerHTML = 'Sugar-AI is thinking<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>';
        
        indicatorElement.appendChild(textElement);
        chatMessages.appendChild(indicatorElement);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // animate the dots
        const dots = textElement.querySelectorAll('.dot');
        let i = 0;
        const dotAnimation = setInterval(() => {
            dots.forEach(dot => dot.style.opacity = '0.2');
            dots[i].style.opacity = '1';
            i = (i + 1) % dots.length;
        }, 300);
        
        return {
            remove: () => {
                clearInterval(dotAnimation);
                indicatorElement.remove();
            }
        };
    }
    
    // handle endpoint selection changes
    function handleEndpointChange() {
        const selectedEndpoint = document.querySelector('input[name="endpoint-choice"]:checked').value;
        if (selectedEndpoint === 'ask-llm-prompted') {
            customPromptOptions.style.display = 'block';
        } else {
            customPromptOptions.style.display = 'none';
        }
    }
    
    // handle preset button clicks
    function applyPreset(presetType) {
        if (!temperatureInput || !topPInput || !repetitionPenaltyInput) return;
        
        switch (presetType) {
            case 'code':
                temperatureInput.value = '0.3';
                topPInput.value = '0.8';
                repetitionPenaltyInput.value = '1.1';
                break;
            case 'creative':
                temperatureInput.value = '0.8';
                topPInput.value = '0.9';
                repetitionPenaltyInput.value = '1.2';
                break;
            case 'factual':
                temperatureInput.value = '0.4';
                topPInput.value = '0.7';
                repetitionPenaltyInput.value = '1.0';
                break;
        }
    }
    
    // get selected endpoint type
    function getSelectedEndpoint() {
        const selectedRadio = document.querySelector('input[name="endpoint-choice"]:checked');
        return selectedRadio ? selectedRadio.value : 'ask';
    }
    
    // send a message to the API
    async function sendMessage(message) {
        addUserMessage(message);
        
        const typingIndicator = showTypingIndicator();
        
        try {
            const selectedEndpoint = getSelectedEndpoint();
            const apiKey = apiKeyField.value;
            let response;
            
            if (selectedEndpoint === 'ask-llm-prompted') {
                // Custom prompt endpoint with JSON body
                const requestBody = {
                    question: message,
                    custom_prompt: customPromptTextarea.value || 'You are a helpful assistant. Provide clear and detailed answers.'
                };
                
                // Add generation parameters if they exist
                if (temperatureInput) requestBody.temperature = parseFloat(temperatureInput.value);
                if (topPInput) requestBody.top_p = parseFloat(topPInput.value);
                if (topKInput) requestBody.top_k = parseInt(topKInput.value);
                if (maxLengthInput) requestBody.max_length = parseInt(maxLengthInput.value);
                if (repetitionPenaltyInput) requestBody.repetition_penalty = parseFloat(repetitionPenaltyInput.value);
                if (truncationInput) requestBody.truncation = truncationInput.checked;
                
                response = await fetch('/ask-llm-prompted', {
                    method: 'POST',
                    headers: {
                        'X-API-Key': apiKey,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody),
                    credentials: 'same-origin'
                });
            } else {
                // Standard endpoints with query parameters
                const endpoint = selectedEndpoint === 'ask' ? '/ask' : '/ask-llm';
                response = await fetch(`${endpoint}?question=${encodeURIComponent(message)}`, {
                    method: 'POST',
                    headers: {
                        'X-API-Key': apiKey
                    },
                    credentials: 'same-origin'
                });
            }
            
            if (!response.ok) {
                throw new Error(`API returned ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // update quota info
            if (data.quota) {
                quotaRemaining.textContent = data.quota.remaining;
                quotaTotal.textContent = data.quota.total;
                
                const quotaPercentage = (data.quota.remaining / data.quota.total) * 100;
                quotaBar.style.width = `${quotaPercentage}%`;
                
                if (quotaPercentage < 20) {
                    quotaBar.style.backgroundColor = '#e74c3c';
                } else if (quotaPercentage < 50) {
                    quotaBar.style.backgroundColor = '#f39c12';
                } else {
                    quotaBar.style.backgroundColor = '#2ecc71';
                }
            }
            
            typingIndicator.remove();
            addAIMessage(data.answer);
            
        } catch (error) {
            typingIndicator.remove();
            addAIMessage(`Error: ${error.message}`);
        }
    }
    
    // event listeners
    if (chatSubmit) {
        chatSubmit.addEventListener('click', () => {
            const message = chatInput.value.trim();
            if (message) {
                sendMessage(message);
                chatInput.value = '';
            }
        });
    }
    
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const message = chatInput.value.trim();
                if (message) {
                    sendMessage(message);
                    chatInput.value = '';
                }
            }
        });
    }
    
    // API key management
    if (toggleApiKeyBtn) {
        toggleApiKeyBtn.addEventListener('click', function() {
            if (apiKeyField.type === 'password') {
                apiKeyField.type = 'text';
                this.textContent = 'Hide Key';
            } else {
                apiKeyField.type = 'password';
                this.textContent = 'Show Key';
            }
        });
    }
    
    if (copyApiKeyBtn) {
        copyApiKeyBtn.addEventListener('click', function() {
            // temporarily change to text to enable copying the actual value
            const originalType = apiKeyField.type;
            apiKeyField.type = 'text';
            apiKeyField.select();
            document.execCommand('copy');
            apiKeyField.type = originalType; // reset to original state
            
            // remove selection to hide the key
            window.getSelection().removeAllRanges();
            
            // show feedback
            const originalText = this.textContent;
            this.textContent = 'Copied!';
            this.style.backgroundColor = getComputedStyle(document.documentElement).getPropertyValue('--accent-color');
            
            setTimeout(() => {
                this.textContent = originalText;
                this.style.backgroundColor = '';
            }, 2000);
        });
    }
    
    // endpoint selection event listeners
    endpointRadios.forEach(radio => {
        radio.addEventListener('change', handleEndpointChange);
    });
    
    // preset button event listeners
    presetBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            applyPreset(this.dataset.preset);
        });
    });
    
    // initialize custom prompt visibility
    handleEndpointChange();
});
