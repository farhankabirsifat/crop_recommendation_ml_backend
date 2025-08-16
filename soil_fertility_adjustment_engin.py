import joblib
import numpy as np
import pandas as pd

# Load all artifacts
artifacts = joblib.load('SFAR_Model/fertility_models_reduced.pkl')
models = artifacts['models']
crop_stats = artifacts['crop_stats']
crop_list = artifacts['crop_list']
# Define adjustment strategies
NUTRIENT_ADJUSTMENTS = {
    'N': {
        'low': "Apply nitrogen-rich fertilizers (urea, ammonium sulfate) at {amount:.1f} kg/ha",
        'high': "Reduce nitrogen fertilizers by {reduction:.0f}%, plant nitrogen-fixing cover crops"
    },
    'P': {
        'low': "Apply phosphorus-rich fertilizers (DAP, rock phosphate) at {amount:.1f} kg/ha",
        'high': "Reduce phosphorus inputs by {reduction:.0f}%, improve drainage"
    },
    'K': {
        'low': "Apply potassium-rich fertilizers (potash) at {amount:.1f} kg/ha",
        'high': "Reduce potassium inputs by {reduction:.0f}%, increase leaching"
    }
}

PH_ADJUSTMENTS = {
    'low': "Apply lime at {lime:.1f} tons/ha to raise pH (target: {target_ph:.1f})",
    'high': "Apply sulfur at {sulfur:.1f} kg/ha to lower pH (target: {target_ph:.1f})"
}

ENVIRONMENT_ADJUSTMENTS = {
    'temperature': {
        'low': "Use greenhouses or row covers to increase temperature by {increase:.1f}°C",
        'high': "Provide shade or plant in cooler seasons to reduce temperature by {reduction:.1f}°C"
    },
    'humidity': {
        'low': "Install irrigation to increase humidity by {increase:.1f}%",
        'high': "Improve drainage to reduce humidity by {reduction:.1f}%"
    }
}


def calculate_adjustment(current, ideal, param):
    """Calculate precise adjustment amounts"""
    diff = ideal - current
    abs_diff = abs(diff)

    if param in ['N', 'P', 'K']:
        # Nutrient adjustment calculations
        amount = abs_diff * 2.5  # kg/ha per unit difference
        reduction = min(100, max(0, (abs_diff / current) * 100)) if current > 0 else 50
        return {
            'amount': amount,
            'reduction': reduction,
            'type': 'low' if diff > 0 else 'high'
        }
    elif param == 'ph':
        # pH adjustment calculations
        return {
            'lime': abs_diff * 2.0,  # tons/ha per pH unit
            'sulfur': abs_diff * 300,  # kg/ha per pH unit
            'target_ph': ideal,
            'type': 'low' if diff > 0 else 'high'
        }
    else:
        # Environmental adjustments
        return {
            'increase': abs_diff if diff > 0 else 0,
            'reduction': abs_diff if diff < 0 else 0,
            'type': 'low' if diff > 0 else 'high'
        }


def get_ml_recommendations(crop, N, P, K, temp, humidity, ph):
    """Get ML-based recommendations with precise quantities"""
    # Predict ideal values using ML models
    # input_data = np.array([[temp, humidity, ph]])
    input_df = pd.DataFrame([[temp, humidity, ph]], columns=['temperature', 'humidity', 'ph'])

    # Get ideal values for nutrients from precomputed stats
    ideal_N = crop_stats[crop]['median']['N']
    ideal_P = crop_stats[crop]['median']['P']
    ideal_K = crop_stats[crop]['median']['K']

    # Predict ideal environmental parameters
    # ideal_temp = models['temperature'][crop].predict(input_data)[0]
    # ideal_humidity = models['humidity'][crop].predict(input_data)[0]
    # ideal_ph = models['ph'][crop].predict(input_data)[0]
    ideal_temp = models['temperature'][crop].predict(input_df)[0]
    ideal_humidity = models['humidity'][crop].predict(input_df)[0]
    ideal_ph = models['ph'][crop].predict(input_df)[0]

    ideal_values = {
        'N': ideal_N,
        'P': ideal_P,
        'K': ideal_K,
        'temperature': ideal_temp,
        'humidity': ideal_humidity,
        'ph': ideal_ph
    }

    recommendations = []

    # Process each parameter
    params = [
        ('N', N), ('P', P), ('K', K),
        ('temperature', temp), ('humidity', humidity), ('ph', ph)
    ]

    for param, current in params:
        ideal = ideal_values[param]

        # Skip if within tolerance
        tolerance = {
            'N': 5, 'P': 3, 'K': 5,
            'temperature': 1, 'humidity': 3, 'ph': 0.3
        }[param]

        if abs(current - ideal) <= tolerance:
            continue

        # Calculate adjustment
        adjustment = calculate_adjustment(current, ideal, param)

        # Generate recommendation
        if param in NUTRIENT_ADJUSTMENTS:
            rec_type = adjustment['type']
            template = NUTRIENT_ADJUSTMENTS[param][rec_type]
            rec = template.format(**adjustment)
            recommendations.append(f"{param}: {rec} (Current: {current:.1f}, Ideal: {ideal:.1f})")
        elif param == 'ph':
            rec_type = adjustment['type']
            template = PH_ADJUSTMENTS[rec_type]
            rec = template.format(**adjustment)
            recommendations.append(f"pH: {rec} (Current: {current:.1f})")
        else:
            rec_type = adjustment['type']
            template = ENVIRONMENT_ADJUSTMENTS[param][rec_type]
            rec = template.format(**adjustment)
            recommendations.append(f"{param}: {rec} (Current: {current:.1f}, Ideal: {ideal:.1f})")

    # Add general advice
    if not recommendations:
        return ["Soil conditions are optimal for this crop!"]

    recommendations.insert(0, f"Precision adjustments for {crop}:")
    recommendations.append("Apply adjustments 4-6 weeks before planting")
    recommendations.append("Retest soil after adjustments")

    return recommendations