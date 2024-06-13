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
        const ytSearchLink = `https://www.youtube.com/results?search_query=${encodeURIComponent(rec.track_name)}+by+${encodeURIComponent(rec.artist_name)}`;
        recDiv.innerHTML = `<a href="${ytSearchLink}" target="_blank"><p><strong>${rec.track_name}</strong> by ${rec.artist_name}</p></a>`;
        recommendationsDiv.appendChild(recDiv);
    });
}

async function fillWithRandomSong() {
    const response = await fetch(`/random`);

    if (!response.ok) {
        throw new Error('Network response was not ok');
    }

    const data = await response.json();

    // set input for songName to song_name in json
    document.getElementById('songName').value = data.track_name;
    // set input for artistName to artist_name in json
    document.getElementById('artistName').value = data.artist_name;

    console.log('Filled with random song:', data);
}