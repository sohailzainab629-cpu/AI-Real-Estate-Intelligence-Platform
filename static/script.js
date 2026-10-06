// ===============================
// PRICE PREDICTION
// ===============================

const predictionForm = document.getElementById("predictionForm");

if (predictionForm) {
    predictionForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const resultBox = document.getElementById("predictionResult");

        resultBox.classList.add("show");
        resultBox.innerHTML = "⏳ Predicting property price...";

        const data = {
            OverallQual: Number(document.getElementById("OverallQual").value),
            GrLivArea: Number(document.getElementById("GrLivArea").value),
            GarageCars: Number(document.getElementById("GarageCars").value),
            GarageArea: Number(document.getElementById("GarageArea").value),
            TotalBsmtSF: Number(document.getElementById("TotalBsmtSF").value),
            firstFlrSF: Number(document.getElementById("firstFlrSF").value),
            FullBath: Number(document.getElementById("FullBath").value),
            TotRmsAbvGrd: Number(document.getElementById("TotRmsAbvGrd").value),
            YearBuilt: Number(document.getElementById("YearBuilt").value),
            YearRemodAdd: Number(document.getElementById("YearRemodAdd").value),
            BedroomAbvGr: Number(document.getElementById("BedroomAbvGr").value),
            Neighborhood: document.getElementById("Neighborhood").value,
            HouseStyle: document.getElementById("HouseStyle").value,
            KitchenQual: document.getElementById("KitchenQual").value
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || "Prediction failed.");
            }

            const price = Number(result.predicted_price).toLocaleString();

            resultBox.innerHTML = `
                <h3>🏠 Predicted Property Price</h3>
                <div class="price-result">$${price}</div>
                <p>Estimated price based on the machine learning model.</p>
            `;

        } catch (error) {
            resultBox.innerHTML = `
                <p>❌ Error: ${error.message}</p>
            `;
        }
    });
}


// ===============================
// PROPERTY RECOMMENDATIONS
// ===============================

const recommendationForm = document.getElementById("recommendationForm");

if (recommendationForm) {
    recommendationForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const resultBox = document.getElementById("recommendationResult");

        resultBox.classList.add("show");
        resultBox.innerHTML = "⏳ Finding suitable properties...";

        const neighborhood =
            document.getElementById("recommendNeighborhood").value;

        const bedroomsValue =
            document.getElementById("bedrooms").value;

        const data = {
            budget: Number(document.getElementById("budget").value),
            neighborhood: neighborhood,
            bedrooms: bedroomsValue
                ? Number(bedroomsValue)
                : null,
            top_n: Number(document.getElementById("topN").value)
        };

        try {
            const response = await fetch("/recommend", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || "Recommendation failed.");
            }

            if (!result.recommendations || result.recommendations.length === 0) {
                resultBox.innerHTML = `
                    <h3>No properties found</h3>
                    <p>Try increasing your budget or changing the filters.</p>
                `;
                return;
            }

            let html = `
                <h3>🏡 Recommended Properties</h3>
                <div class="recommendations">
            `;

            result.recommendations.forEach((property) => {
                html += `
                    <div class="property-card">
                        <h3>Property #${property.Id}</h3>

                        <p>
                            <strong>Price:</strong>
                            $${Number(property.SalePrice).toLocaleString()}
                        </p>

                        <p>
                            <strong>Location:</strong>
                            ${property.Neighborhood}
                        </p>

                        <p>
                            <strong>Bedrooms:</strong>
                            ${property.BedroomAbvGr}
                        </p>

                        <p>
                            <strong>Bathrooms:</strong>
                            ${property.FullBath}
                        </p>

                        <p>
                            <strong>Living Area:</strong>
                            ${property.GrLivArea} sq ft
                        </p>

                        <p>
                            <strong>Overall Quality:</strong>
                            ${property.OverallQual}/10
                        </p>
                    </div>
                `;
            });

            html += "</div>";

            resultBox.innerHTML = html;

        } catch (error) {
            resultBox.innerHTML = `
                <p>❌ Error: ${error.message}</p>
            `;
        }
    });
}


// ===============================
// AI REAL ESTATE ASSISTANT
// ===============================

const chatForm = document.getElementById("chatForm");
const chatMessages = document.getElementById("chatMessages");

if (chatForm) {
    chatForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const questionInput =
            document.getElementById("chatQuestion");

        const question = questionInput.value.trim();

        if (!question) {
            return;
        }

        // Show user message
        addChatMessage(question, "user-message");

        questionInput.value = "";

        // Loading message
        const loadingMessage = document.createElement("div");
        loadingMessage.className = "message ai-message";
        loadingMessage.innerHTML = "⏳ Thinking...";
        chatMessages.appendChild(loadingMessage);

        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question
                })
            });

            const result = await response.json();

            loadingMessage.remove();

            if (!response.ok) {
                throw new Error(result.detail || "AI request failed.");
            }

            addChatMessage(
                result.answer,
                "ai-message"
            );

        } catch (error) {
            loadingMessage.remove();

            addChatMessage(
                "❌ Error: " + error.message,
                "ai-message"
            );
        }
    });
}


// ===============================
// CHAT MESSAGE FUNCTION
// ===============================

function addChatMessage(message, className) {

    const messageElement = document.createElement("div");

    messageElement.className =
        "message " + className;

    // Preserve line breaks
    messageElement.innerHTML =
        message.replace(/\n/g, "<br>");

    chatMessages.appendChild(messageElement);

    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}