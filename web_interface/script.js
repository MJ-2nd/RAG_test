// Global variables
let currentFunctionCalls = [];
let executionResultsArray = [];

// DOM elements
const userInput = document.getElementById('userInput');
const sendButton = document.getElementById('sendButton');
const loadingIndicator = document.getElementById('loadingIndicator');
const responseSection = document.getElementById('responseSection');
const llmAnswer = document.getElementById('llmAnswer');
const functionCallsSection = document.getElementById('functionCallsSection');
const functionCallsList = document.getElementById('functionCallsList');
const executionResults = document.getElementById('executionResults');
const resultsList = document.getElementById('resultsList');
const statusText = document.getElementById('statusText');
const deviceCount = document.getElementById('deviceCount');

// API endpoints
const API_BASE = 'http://localhost:8000';
const GENERATE_ENDPOINT = `${API_BASE}/generate`;
const EXECUTE_FUNCTION_ENDPOINT = `${API_BASE}/execute_function`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
});

function initializeApp() {
    updateStatus('Initializing...');
    checkServerHealth();
    updateDeviceCount();
}

function setupEventListeners() {
    // Send button click
    sendButton.addEventListener('click', handleSendClick);
    
    // Enter key in textarea
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleSendClick();
        }
    });
    
    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    });
}

async function checkServerHealth() {
    try {
        const response = await fetch(HEALTH_ENDPOINT);
        if (response.ok) {
            const data = await response.json();
            updateStatus('Connected to LLM Server');
            console.log('Server health:', data);
        } else {
            updateStatus('Server connection failed');
        }
    } catch (error) {
        updateStatus('Server not available');
        console.error('Health check failed:', error);
    }
}

async function updateDeviceCount() {
    try {
        const response = await fetch(`${API_BASE}/execute_function`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                function_name: 'get_connected_devices',
                arguments: {}
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            const devices = data.result || [];
            deviceCount.textContent = devices.length;
        }
    } catch (error) {
        console.error('Failed to get device count:', error);
        deviceCount.textContent = '?';
    }
}

async function handleSendClick() {
    const input = userInput.value.trim();
    if (!input) {
        alert('Please enter a message');
        return;
    }
    
    // Disable send button and show loading
    sendButton.disabled = true;
    showLoading(true);
    updateStatus('Generating response...');
    
    try {
        const response = await generateLLMResponse(input);
        console.log('Response:', response);
        displayResponse(response);
    } catch (error) {
        console.error('Error generating response:', error);
        updateStatus('Error generating response');
        alert('Failed to generate response. Please try again.');
    } finally {
        sendButton.disabled = false;
        showLoading(false);
    }
}

async function generateLLMResponse(userInput) {
    const response = await fetch(GENERATE_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            query: userInput,
            max_tokens: 1000,
            temperature: 0.7
        })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
}

function displayResponse(response) {
    // Display LLM answer
    llmAnswer.textContent = response.response;
    responseSection.classList.remove('hidden');
    
    // Display function calls if any
    if (response.function_calls && response.function_calls.length > 0) {
        currentFunctionCalls = response.function_calls;
        displayFunctionCalls(response.function_calls);
        functionCallsSection.classList.remove('hidden');
        updateStatus(`${response.function_calls.length} function(s) ready to execute`);
    } else {
        functionCallsSection.classList.add('hidden');
        updateStatus('Response generated successfully');
    }
    
    // Clear previous execution results
    executionResults.classList.add('hidden');
    resultsList.innerHTML = '';
}

function displayFunctionCalls(functionCalls) {
    functionCallsList.innerHTML = '';
    
    functionCalls.forEach((funcCall, index) => {
        const functionBlock = createFunctionBlock(funcCall, index);
        functionCallsList.appendChild(functionBlock);
    });
}

function createFunctionBlock(funcCall, index) {
    const block = document.createElement('div');
    block.className = 'function-call-block';
    block.dataset.index = index;
    
    const functionName = funcCall.name || 'Unknown Function';
    const arguments_ = funcCall.arguments || {};
    
    block.innerHTML = `
        <div class="function-header">
            <div class="function-name">${functionName}</div>
            <div class="function-status status-pending">Pending</div>
        </div>
        <div class="function-details">
            <div class="function-description">Function call #${index + 1}</div>
            <div class="function-arguments">
                ${Object.entries(arguments_).map(([key, value]) => `
                    <div class="argument-item">
                        <span class="argument-name">${key}:</span>
                        <span class="argument-value">${JSON.stringify(value)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
        <div class="function-actions">
            <button class="execute-btn" onclick="executeFunction(${index})">
                🚀 Execute
            </button>
        </div>
        <div class="function-result" style="display: none;"></div>
    `;
    
    return block;
}

async function executeFunction(index) {
    const functionBlock = document.querySelector(`[data-index="${index}"]`);
    const executeBtn = functionBlock.querySelector('.execute-btn');
    const statusDiv = functionBlock.querySelector('.function-status');
    const resultDiv = functionBlock.querySelector('.function-result');
    
    // Update UI to show executing state
    executeBtn.disabled = true;
    executeBtn.textContent = '⏳ Executing...';
    statusDiv.textContent = 'Executing';
    statusDiv.className = 'function-status status-executing';
    resultDiv.style.display = 'none';
    
    updateStatus(`Executing function ${index + 1}...`);
    
    try {
        const funcCall = currentFunctionCalls[index];
        const response = await fetch(EXECUTE_FUNCTION_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                function_name: funcCall.name,
                arguments: funcCall.arguments || {}
            })
        });
        
        const result = await response.json();
        
        // Display result
        resultDiv.style.display = 'block';
        if (result.error) {
            resultDiv.className = 'function-result result-error';
            resultDiv.textContent = `Error: ${result.error}`;
            statusDiv.textContent = 'Error';
            statusDiv.className = 'function-status status-error';
        } else {
            resultDiv.className = 'function-result result-success';
            resultDiv.textContent = `Success: ${JSON.stringify(result.result, null, 2)}`;
            statusDiv.textContent = 'Success';
            statusDiv.className = 'function-status status-success';
        }
        
        // Add to execution results
        addExecutionResult(funcCall.name, result);
        
        updateStatus(`Function ${index + 1} executed successfully`);
        
    } catch (error) {
        console.error('Function execution failed:', error);
        
        resultDiv.style.display = 'block';
        resultDiv.className = 'function-result result-error';
        resultDiv.textContent = `Error: ${error.message}`;
        statusDiv.textContent = 'Error';
        statusDiv.className = 'function-status status-error';
        
        updateStatus(`Function ${index + 1} execution failed`);
    } finally {
        executeBtn.disabled = false;
        executeBtn.textContent = '🚀 Execute';
    }
}

function addExecutionResult(functionName, result) {
    executionResults.classList.remove('hidden');
    
    const resultItem = document.createElement('div');
    resultItem.className = `result-item ${result.error ? 'error' : 'success'}`;
    
    const timestamp = new Date().toLocaleTimeString();
    resultItem.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 5px;">
            ${functionName} - ${timestamp}
        </div>
        <div style="font-size: 14px;">
            ${result.error ? `Error: ${result.error}` : `Success: ${JSON.stringify(result.result, null, 2)}`}
        </div>
    `;
    
    resultsList.appendChild(resultItem);
    resultsList.scrollTop = resultsList.scrollHeight;
}

function showLoading(show) {
    if (show) {
        loadingIndicator.classList.remove('hidden');
        responseSection.classList.add('hidden');
        functionCallsSection.classList.add('hidden');
        executionResults.classList.add('hidden');
    } else {
        loadingIndicator.classList.add('hidden');
    }
}

function updateStatus(message) {
    statusText.textContent = message;
    console.log('Status:', message);
}

// Utility functions
function formatJson(obj) {
    return JSON.stringify(obj, null, 2);
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    updateStatus('An error occurred');
});

// Network status monitoring
window.addEventListener('online', function() {
    updateStatus('Back online');
    checkServerHealth();
});

window.addEventListener('offline', function() {
    updateStatus('Network disconnected');
}); 