// ==========================================
// API CONFIGURATION
// ==========================================

const API_BASE_URL = "http://127.0.0.1:8000";


// ==========================================
// GET RECOMMENDATIONS
// ==========================================

async function getRecommendations() {

    const userIdInput =
        document.getElementById("userId");

    const numberOfMovies =
        document.getElementById("numberOfMovies");

    const button =
        document.getElementById("recommendButton");

    const buttonText =
        document.getElementById("buttonText");

    const spinner =
        document.getElementById("loadingSpinner");

    const errorMessage =
        document.getElementById("errorMessage");

    const resultsSection =
        document.getElementById("resultsSection");

    const emptyState =
        document.getElementById("emptyState");

    const grid =
        document.getElementById("recommendationGrid");

    const displayUserId =
        document.getElementById("displayUserId");


    // ======================================
    // READ INPUT
    // ======================================

    const userId =
        parseInt(userIdInput.value);

    const n =
        parseInt(numberOfMovies.value);


    // ======================================
    // VALIDATE USER ID
    // ======================================

    if (!userId || userId < 1) {

        showError(
            "Please enter a valid User ID."
        );

        return;
    }


    // ======================================
    // RESET UI
    // ======================================

    errorMessage.classList.add("hidden");

    resultsSection.classList.add("hidden");

    grid.innerHTML = "";

    emptyState.classList.add("hidden");


    // ======================================
    // LOADING STATE
    // ======================================

    button.disabled = true;

    buttonText.textContent =
        "Loading...";

    spinner.classList.remove("hidden");


    try {

        // ==================================
        // API REQUEST
        // ==================================

        const url =
            `${API_BASE_URL}/recommendations/${userId}?n=${n}`;


        console.log(
            "Requesting:",
            url
        );


        const response =
            await fetch(url);


        // ==================================
        // CHECK HTTP STATUS
        // ==================================

        if (!response.ok) {

            let errorText =
                `API returned status ${response.status}.`;

            try {

                const errorData =
                    await response.json();

                if (errorData.detail) {

                    errorText =
                        errorData.detail;
                }

            } catch (e) {

                // Response wasn't JSON.
            }

            throw new Error(errorText);
        }


        // ==================================
        // PARSE JSON
        // ==================================

        const data =
            await response.json();


        console.log(
            "API response:",
            data
        );


        // ==================================
        // HANDLE RESPONSE
        // ==================================

        displayUserId.textContent =
            userId;


        renderRecommendations(
            data
        );


        resultsSection.classList.remove(
            "hidden"
        );


    } catch (error) {

        console.error(
            "Recommendation error:",
            error
        );


        showError(
            "Could not load recommendations. " +
            error.message
        );


        emptyState.classList.remove(
            "hidden"
        );


    } finally {

        // ==================================
        // RESTORE BUTTON
        // ==================================

        button.disabled = false;

        buttonText.textContent =
            "Get Recommendations";

        spinner.classList.add(
            "hidden"
        );

    }
}


// ==========================================
// RENDER RECOMMENDATIONS
// ==========================================

function renderRecommendations(data) {

    const grid =
        document.getElementById(
            "recommendationGrid"
        );


    grid.innerHTML = "";


    // ======================================
    // DETERMINE RECOMMENDATION ARRAY
    // ======================================

    let recommendations = [];


    /*
       Your API may return the recommendations
       directly as an array, or inside a field
       such as "recommendations".
    */


    if (Array.isArray(data)) {

        recommendations =
            data;

    } else if (
        data &&
        Array.isArray(data.recommendations)
    ) {

        recommendations =
            data.recommendations;

    } else if (
        data &&
        Array.isArray(data.results)
    ) {

        recommendations =
            data.results;

    }


    // ======================================
    // NO RESULTS
    // ======================================

    if (recommendations.length === 0) {

        grid.innerHTML = `

            <div class="movie-card">

                <div class="movie-title">
                    No recommendations found.
                </div>

                <p style="
                    color:#999;
                    margin-top:10px;
                ">
                    Try another User ID.
                </p>

            </div>

        `;

        return;
    }


    // ======================================
    // CREATE MOVIE CARDS
    // ======================================

    recommendations.forEach(
        (movie, index) => {

            const card =
                createMovieCard(
                    movie,
                    index
                );

            grid.appendChild(card);

        }
    );
}


// ==========================================
// CREATE MOVIE CARD
// ==========================================

function createMovieCard(
    movie,
    index
) {

    const card =
        document.createElement("article");


    card.className =
        "movie-card";


    // ======================================
    // GET MOVIE TITLE
    // ======================================

    const title =
        movie.title ||
        movie.movie_title ||
        movie.name ||
        "Unknown Movie";


    // ======================================
    // GET SCORES
    // ======================================

    const hybrid =
    getNumber(
        movie.hybridScore,
        movie.hybrid_score,
        movie.hybrid,
        movie.score
    );

const collaborative =
    getNumber(
        movie.collaborativeScore,
        movie.collaborative,
        movie.collaborative_score,
        movie.predicted_rating
    );

const content =
    getNumber(
        movie.contentScore,
        movie.content,
        movie.content_score,
        movie.similarity
    );

const quality =
    getNumber(
        movie.qualityScore,
        movie.quality,
        movie.quality_score
    );

    // ======================================
    // FORMAT SCORES
    // ======================================

    const hybridText =
        hybrid !== null
            ? hybrid.toFixed(3)
            : "—";


    const collaborativeText =
        collaborative !== null
            ? collaborative.toFixed(2)
            : "—";


    const contentText =
        content !== null
            ? content.toFixed(3)
            : "—";


    const qualityText =
        quality !== null
            ? quality.toFixed(3)
            : "—";


    // ======================================
    // CARD HTML
    // ======================================

    card.innerHTML = `

        <div class="movie-number">
            ${index + 1}
        </div>


        <div class="movie-info">

            <div class="movie-title">
                ${escapeHtml(title)}
            </div>


            <div class="movie-score">
                ⭐ Hybrid Score:
                ${hybridText}
            </div>


            <div class="score-details">

                <div class="score-item">

                    <span class="score-label">
                        Collaborative
                    </span>

                    <span class="score-value">
                        ${collaborativeText}
                    </span>

                </div>


                <div class="score-item">

                    <span class="score-label">
                        Content
                    </span>

                    <span class="score-value">
                        ${contentText}
                    </span>

                </div>


                <div class="score-item">

                    <span class="score-label">
                        Quality
                    </span>

                    <span class="score-value">
                        ${qualityText}
                    </span>

                </div>

            </div>

        </div>

    `;


    return card;
}


// ==========================================
// GET NUMBER SAFELY
// ==========================================

function getNumber(...values) {

    for (
        const value of values
    ) {

        if (
            value !== undefined &&
            value !== null &&
            value !== "" &&
            !isNaN(Number(value))
        ) {

            return Number(value);

        }

    }

    return null;
}


// ==========================================
// ESCAPE HTML
// ==========================================

function escapeHtml(value) {

    return String(value)

        .replace(
            /&/g,
            "&amp;"
        )

        .replace(
            /</g,
            "&lt;"
        )

        .replace(
            />/g,
            "&gt;"
        )

        .replace(
            /"/g,
            "&quot;"
        )

        .replace(
            /'/g,
            "&#039;"
        );
}


// ==========================================
// SHOW ERROR
// ==========================================

function showError(message) {

    const errorMessage =
        document.getElementById(
            "errorMessage"
        );


    errorMessage.textContent =
        message;


    errorMessage.classList.remove(
        "hidden"
    );
}


// ==========================================
// ENTER KEY SUPPORT
// ==========================================

document
    .getElementById("userId")
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter"
            ) {

                getRecommendations();

            }

        }
    );