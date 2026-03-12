/**
 * INTELLICORE - Main JavaScript
 * Curated AI Personas Platform
 */

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Intellicore Platform Loaded');
    
    // Highlight current page in navigation
    highlightCurrentPage();
    
    // Initialize suggested questions if on chat page
    if (document.querySelector('.suggestion-btn')) {
        initSuggestedQuestions();
    }
});

/**
 * Highlight current page in navigation
 */
function highlightCurrentPage() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        const linkPage = link.getAttribute('href');
        link.classList.remove('active');
        
        if (linkPage === currentPage || 
            (currentPage === '' && linkPage === 'index.html')) {
            link.classList.add('active');
        }
    });
}

/**
 * Initialize suggested question buttons
 */
function initSuggestedQuestions() {
    const suggestionBtns = document.querySelectorAll('.suggestion-btn');
    
    suggestionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const question = this.getAttribute('data-question');
            if (question) {
                populateMessageInput(question);
            }
        });
    });
}

/**
 * Populate message input with suggested question
 */
function populateMessageInput(text) {
    const messageInput = document.getElementById('messageInput');
    if (messageInput) {
        messageInput.value = text;
        messageInput.focus();
    }
}

/**
 * Format timestamp
 */
function formatTimestamp(date = new Date()) {
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes.toString().padStart(2, '0');
    
    return `${displayHours}:${displayMinutes} ${ampm}`;
}

/**
 * Scroll to bottom of container
 */
function scrollToBottom(element) {
    if (element) {
        element.scrollTop = element.scrollHeight;
    }
}

/**
 * Close modal
 */
function closeModal() {
    const modal = document.getElementById('creditsModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

/**
 * Show modal
 */
function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'flex';
    }
}

// Export functions for use in other scripts
window.IntellicoreApp = {
    formatTimestamp,
    scrollToBottom,
    closeModal,
    showModal,
    populateMessageInput
};
