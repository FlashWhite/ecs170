document.getElementById('songForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const songName = document.getElementById('songName').value;
    const artistName = document.getElementById('artistName').value;
    const model = document.getElementById('model').value;

    console.log('Sending request with:', { songName, artistName, model });

    try {
        const response = await fetch(`http://127.0.0.1:5000/recommend`, {
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
    } catch (error) {
        console.error('Fetch error:', error);
    }
});

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
