import pickel
import numpy as np

# Load saved model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Example new house input
# [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
new_house = np.array([[8.3, 40, 6.5, 1.1, 1000, 3.0, 37.8, -122.4]])

prediction = model.predict(new_house)

print("Predicted House Price:", prediction[0])