document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('analyze-form');
    const input = document.getElementById('youtube-url');
    const btn = document.getElementById('submit-btn');
    const errorMsg = document.getElementById('error-message');
    
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultsState = document.getElementById('results-state');
    
    const statPos = document.getElementById('stat-positive');
    const statNeg = document.getElementById('stat-negative');
    const statNeu = document.getElementById('stat-neutral');
    const totalCount = document.getElementById('total-comments');
    const commentList = document.getElementById('comment-list');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        let url = input.value.trim();
        if (!validateUrl(url)) {
            showError("Please enter a valid YouTube URL");
            return;
        }
        
        hideError();
        
        // UI states handling
        btn.disabled = true;
        input.disabled = true;
        const originalBtnText = btn.textContent;
        btn.textContent = 'ANALYZING...';
        
        emptyState.classList.add('hidden');
        resultsState.classList.add('hidden');
        loadingState.classList.remove('hidden');
        commentList.innerHTML = '';
        
        try {
            const res = await fetch('/analyze-video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            
            const data = await res.json();
            
            if (!res.ok) {
                throw new Error(data.error || "An unknown error occurred");
            }
            
            // Populate stats
            statPos.textContent = data.positive;
            statNeg.textContent = data.negative;
            statNeu.textContent = data.neutral;
            totalCount.textContent = `TOTAL: ${data.total}`;
            
            renderComments(data.results);
            
            loadingState.classList.add('hidden');
            resultsState.classList.remove('hidden');
            
            setTimeout(() => {
                resultsState.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
            
        } catch (err) {
            loadingState.classList.add('hidden');
            if (commentList.children.length === 0) {
                emptyState.classList.remove('hidden');
            } else {
                resultsState.classList.remove('hidden');
                resultsState.style.opacity = '0.5';
            }
            showError(err.message);
        } finally {
            btn.disabled = false;
            input.disabled = false;
            btn.textContent = 'ANALYZE';
        }
    });

    function validateUrl(url) {
        if (!url) return false;
        try {
            const parsed = new URL(url);
            return (
                (parsed.hostname.includes('youtube.com') && (parsed.pathname === '/watch' || parsed.pathname.startsWith('/shorts/'))) ||
                parsed.hostname.includes('youtu.be')
            );
        } catch (e) {
            return false;
        }
    }
    
    function showError(msg) {
        errorMsg.textContent = msg;
        errorMsg.classList.remove('hidden');
    }
    
    function hideError() {
        errorMsg.textContent = '';
        errorMsg.classList.add('hidden');
        resultsState.style.opacity = '1';
    }
    
    function renderComments(results) {
        results.forEach(item => {
            const card = document.createElement('div');
            let type = 'neutral';
            if (item.sentiment === 1) type = 'positive';
            else if (item.sentiment === -1) type = 'negative';
            
            card.className = `comment-card border-[3px] border-black p-5 border-l-8 border-l-${type} bg-white shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] flex flex-col h-full`;
            
            const header = document.createElement('div');
            header.className = 'mb-3 flex justify-between items-start';
            
            const badge = document.createElement('span');
            badge.className = `badge-${type} px-2 py-1 text-xs font-mono uppercase font-bold border-2 border-black`;
            badge.textContent = type;
            
            header.appendChild(badge);
            
            const textContainer = document.createElement('div');
            textContainer.className = 'font-mono text-sm leading-relaxed mb-4 flex-grow break-words text-black';
            
            const fullText = item.comment;
            const threshold = 180;
            
            if (fullText.length > threshold) {
                const shortText = fullText.slice(0, threshold) + '...';
                textContainer.innerHTML = `<span class="comment-text">${escapeHtml(shortText)}</span>`;
                
                const toggleLink = document.createElement('a');
                toggleLink.href = '#';
                toggleLink.className = 'text-blue-700 hover:text-black hover:underline font-bold mt-2 block text-xs uppercase';
                toggleLink.textContent = 'Show more';
                
                let expanded = false;
                toggleLink.addEventListener('click', (e) => {
                    e.preventDefault();
                    expanded = !expanded;
                    if (expanded) {
                        textContainer.innerHTML = `<span class="comment-text">${escapeHtml(fullText)}</span>`;
                        toggleLink.textContent = 'Show less';
                    } else {
                        textContainer.innerHTML = `<span class="comment-text">${escapeHtml(shortText)}</span>`;
                        toggleLink.textContent = 'Show more';
                    }
                });
                
                card.appendChild(header);
                card.appendChild(textContainer);
                card.appendChild(toggleLink);
            } else {
                textContainer.textContent = fullText;
                card.appendChild(header);
                card.appendChild(textContainer);
            }
            
            commentList.appendChild(card);
        });
    }
    
    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
