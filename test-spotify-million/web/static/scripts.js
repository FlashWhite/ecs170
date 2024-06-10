document.getElementById('songForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const songName = document.getElementById('songName').value;
    const artistName = document.getElementById('artistName').value;
    const model = document.getElementById('model').value;

    console.log('Sending request with:', { songName, artistName, model });

    try {
        displayLoading(true);
        const response = await fetch(`/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ songName, artistName, model })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log('Received recommendations:', data);
        displayRecommendations(data);
        displayLoading(false);
    } catch (error) {
        console.error('Fetch error:', error);
        displayLoading(false);
    }
});

function displayLoading(show) {
    const loadingDiv = document.getElementById('loading');
    const submitButton = document.getElementById('submit');
    if(show) {
        // display loading text
        loadingDiv.style.display = 'block';
        loadingDiv.innerHTML = 'Loading...';

        submitButton.disabled = true;
    } else {
        loadingDiv.style.display = 'none';
        loadingDiv.innerHTML = '';

        submitButton.disabled = false;
    }
}

function displayRecommendations(recommendations) {
    const recommendationsDiv = document.getElementById('recommendations');
    recommendationsDiv.innerHTML = '';

    if (recommendations.length === 0) {
        recommendationsDiv.innerHTML = '<p>No recommendations found.</p>';
        return;
    }

    recommendations.forEach(rec => {
        const recDiv = document.createElement('div');
        recDiv.className = 'recommendation';
        recDiv.innerHTML = `<p><strong>${rec.track_name}</strong> by ${rec.artist_name}</p>`;
        recommendationsDiv.appendChild(recDiv);
    });
}
