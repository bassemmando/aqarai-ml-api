from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
import os
app = Flask(__name__)
app.json.ensure_ascii = False
app = Flask(__name__)
app.json.ensure_ascii = False

print("🔥 APP INITIALIZED")
print("🔥 ROUTES:", app.url_map)
# 👇 حط دول هنا مباشرة
print("🔥 STARTING APP IMPORT")
print("🔥 ROUTES:", app.url_map)

# ============================================
# تحميل الموديل
# ============================================
model        = joblib.load('model/price_model.pkl')
le_city      = joblib.load('model/le_city.pkl')
le_finishing = joblib.load('model/le_finishing.pkl')

# ============================================
# تحميل Dataset التوصيات
# ============================================
df = pd.read_csv('rec_model/apartments.csv', encoding='utf-8-sig')

print("✅ كل حاجة محملة وجاهزة!")

# ============================================
# المدن والتشطيب
# ============================================
CITIES = [
    "القاهرة","الجيزة","الإسكندرية","العاصمة الإدارية",
    "القاهرة الجديدة","الساحل الشمالي","الغردقة","شرم الشيخ","العين السخنة"
]
FINISHING = ["Unfinished","Semi-Finished","Fully Finished","Super Lux"]

# ============================================
# Helper: Listing Status
# ============================================
def get_listing_status(actual_price, predicted_price):
    if actual_price is None:
        return None
    diff_pct = ((actual_price - predicted_price) / predicted_price) * 100
    if diff_pct > 15:
        return {"status":"overpriced","label":"Overpriced","color":"yellow",
                "message":"This listing is priced above the market average.",
                "difference_percentage": round(diff_pct, 1)}
    elif diff_pct < -15:
        return {"status":"underpriced","label":"Underpriced","color":"blue",
                "message":"This listing is priced below the market average.",
                "difference_percentage": round(diff_pct, 1)}
    else:
        return {"status":"fair_price","label":"Fair Price","color":"green",
                "message":"This listing is priced fairly according to our AI analysis.",
                "difference_percentage": round(diff_pct, 1)}

# ============================================
# Helper: Recommendations
# ============================================
def get_similar_apartments(apartment_id=None, city=None, area=None,
                            rooms=None, bathrooms=None,
                            finishing=None, price=None, top_n=3):
    if apartment_id:
        apt = df[df['apartment_id'] == apartment_id]
        if len(apt) == 0:
            return []
        apt = apt.iloc[0]
        city      = apt['city']
        area      = apt['area_sqm']
        rooms     = apt['number_of_rooms']
        bathrooms = apt['number_of_bathrooms']
        finishing = apt['finishing_status']
        price     = apt['price_egp']

    candidates = df.copy()
    if apartment_id:
        candidates = candidates[candidates['apartment_id'] != apartment_id]

    scores = []
    for _, row in candidates.iterrows():
        score = 0
        if row['city'] == city: score += 40
        if price and price > 0:
            price_diff = abs(row['price_egp'] - price) / price
            if price_diff <= 0.10: score += 30
            elif price_diff <= 0.20: score += 20
            elif price_diff <= 0.30: score += 10
        room_diff = abs(row['number_of_rooms'] - rooms)
        if room_diff == 0: score += 20
        elif room_diff == 1: score += 10
        if row['finishing_status'] == finishing: score += 10
        scores.append(score)

    candidates = candidates.copy()
    candidates['match_score'] = scores
    top = candidates.sort_values('match_score', ascending=False).head(top_n)

    results = []
    for _, row in top.iterrows():
        results.append({
            'apartment_id':        row['apartment_id'],
            'city':                row['city'],
            'area_sqm':            int(row['area_sqm']),
            'number_of_rooms':     int(row['number_of_rooms']),
            'number_of_bathrooms': int(row['number_of_bathrooms']),
            'finishing_status':    row['finishing_status'],
            'price_egp':           int(row['price_egp']),
            'match_score':         int(row['match_score'])
        })
    return results

# ============================================
# Routes
# ============================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "AqarAI API is running!"})

# --- Price Prediction ---
@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        required = ['city','floor_number','area_sqm',
                    'number_of_rooms','number_of_bathrooms','finishing_status']
        for field in required:
            if field not in data:
                return jsonify({"success":False,"error":f"Missing field: {field}"}), 400

        city             = data['city']
        floor_number     = int(data['floor_number'])
        area_sqm         = int(data['area_sqm'])
        number_of_rooms  = int(data['number_of_rooms'])
        number_of_baths  = int(data['number_of_bathrooms'])
        finishing_status = data['finishing_status']
        actual_price     = data.get('actual_price', None)

        if city not in CITIES:
            return jsonify({"success":False,"error":"Invalid city"}), 400
        if finishing_status not in FINISHING:
            return jsonify({"success":False,"error":"Invalid finishing"}), 400

        city_enc      = le_city.transform([city])[0]
        finishing_enc = le_finishing.transform([finishing_status])[0]

        input_data = pd.DataFrame([{
            'floor_number':        floor_number,
            'area_sqm':            area_sqm,
            'number_of_rooms':     number_of_rooms,
            'number_of_bathrooms': number_of_baths,
            'city_enc':            city_enc,
            'finishing_enc':       finishing_enc,
            'rooms_bath_total':    number_of_rooms + number_of_baths,
            'area_per_room':       area_sqm / number_of_rooms
        }])

        predicted      = int(model.predict(input_data)[0])
        listing_status = get_listing_status(actual_price, predicted)

        return jsonify({
            "success":         True,
            "predicted_price": predicted,
            "min_price":       int(predicted * 0.90),
            "max_price":       int(predicted * 1.10),
            "price_per_sqm":   int(predicted / area_sqm),
            "listing_status":  listing_status
        }), 200

    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

# --- Options ---
@app.route('/api/options', methods=['GET'])
def get_options():
    return jsonify({"cities":CITIES,"finishing":FINISHING}), 200

# --- Similar by ID ---
@app.route('/api/similar/<apartment_id>', methods=['GET'])
def similar_by_id(apartment_id):
    try:
        top_n   = int(request.args.get('top_n', 3))
        results = get_similar_apartments(apartment_id=apartment_id, top_n=top_n)
        if not results:
            return jsonify({"success":False,"error":f"Apartment {apartment_id} not found"}), 404
        return jsonify({
            "success":            True,
            "apartment_id":       apartment_id,
            "similar_apartments": results,
            "total":              len(results)
        }), 200
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

# --- Similar by Features ---
@app.route('/api/similar', methods=['POST'])
def similar_by_features():
    try:
        data     = request.get_json()
        required = ['city','area_sqm','number_of_rooms',
                    'number_of_bathrooms','finishing_status','price_egp']
        for field in required:
            if field not in data:
                return jsonify({"success":False,"error":f"Missing field: {field}"}), 400

        top_n   = int(data.get('top_n', 3))
        results = get_similar_apartments(
            city      = data['city'],
            area      = int(data['area_sqm']),
            rooms     = int(data['number_of_rooms']),
            bathrooms = int(data['number_of_bathrooms']),
            finishing = data['finishing_status'],
            price     = int(data['price_egp']),
            top_n     = top_n
        )
        return jsonify({
            "success":            True,
            "similar_apartments": results,
            "total":              len(results)
        }), 200
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 500

# ============================================
# Run
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
