// app.js
console.log("JavaScript is working!");

fetch("https://crop-recommendation-ml-backend-2.onrender.com/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    N: 90,
    P: 42,
    K: 43,
    temperature: 22,
    humidity: 80,
    ph: 6.5
  })
})
.then(res => res.json())
.then(data => {
  if (data.recommendations && Array.isArray(data.recommendations)) {
    data.recommendations.forEach(rec => {
      console.log(`Crop: ${rec.crop}, Probability: ${rec.probability}`);
    });
  } else {
    console.log("No recommendations found in response.");
  }
})
.catch(err => console.error("Error:", err));