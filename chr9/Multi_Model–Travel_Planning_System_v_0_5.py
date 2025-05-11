"""
Travel Planning System with Multi-Agent Architecture

This script defines multiple specialized agents, each designed for a specific aspect of travel planning:

1. WeatherAnalysisAgent:
   - Predicts the best months to visit a location using historical weather data.
   - Uses a Random Forest regression model for predictions.

2. HotelRecommenderAgent:
   - Recommends hotels based on semantic similarity between user preferences and hotel descriptions.
   - Utilizes SentenceTransformer embeddings for semantic comparison.

3. ItineraryPlannerAgent:
   - Generates detailed travel itineraries using a text-generation language model.
   - Leverages the GPT-2 model for generating realistic itineraries based on provided prompts.

4. SummaryAgent:
   - Summarizes the travel plan details into a concise, client-friendly email.
   - Estimates the total cost of the trip and uses GPT-2 to generate the summary content.

These agents collectively enable an automated and intelligent approach to personalized travel planning.
"""
from typing import List, Dict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
from transformers import pipeline


class WeatherAnalysisAgent:
    """
    An agent that analyzes historical weather data to predict optimal travel periods.

    Attributes:
        model (RandomForestRegressor): A random forest model trained on historical weather data.
    """
    def __init__(self):
        # Uses RandomForest for weather predictions based on historical data
        self.model = RandomForestRegressor(n_estimators=100)

    def train(self, historical_data: Dict):
        # Training on historical weather data
        X = np.array([[d['month'], d['latitude'], d['longitude']] for d in historical_data])
        y = np.array([d['weather_score'] for d in historical_data])
        self.model.fit(X, y)

    def predict_best_time(self, location: Dict) -> Dict:
        # Predicts the best time to visit a location based on weather patterns
        predictions = []
        for month in range(1, 13):
            # predict returns a 2D array, we take the first (and only) element
            prediction = self.model.predict([[
                month,
                location['latitude'],
                location['longitude']
            ]]).item()  # .item() converts numpy array to scalar
            predictions.append({'month': month, 'score': float(prediction)})

        return {
            'best_months': sorted(predictions, key=lambda x: x['score'], reverse=True)[:3],
            'location': location
        }


class HotelRecommenderAgent:
    """
    An agent for recommending hotels that best match user preferences using semantic embeddings.

    Attributes:
        encoder (SentenceTransformer): Sentence embedding model for semantic matching.
        hotels_db (List[Dict]): Database of hotel information.
        hotels_embeddings (np.ndarray): Precomputed embeddings of hotel descriptions.
    """
    def __init__(self):
        # Uses SentenceTransformer for hotel description embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.hotels_db = []
        self.hotels_embeddings = None

    def add_hotels(self, hotels: List[Dict]):
        # Updates the hotel database and computes embeddings
        self.hotels_db = hotels
        descriptions = [h['description'] for h in hotels]
        self.hotels_embeddings = self.encoder.encode(descriptions)

    def find_hotels(self, preferences: str, top_k: int = 5) -> List[Dict]:
        # Finds hotels most similar to preferences using semantic similarity
        pref_embedding = self.encoder.encode([preferences])
        similarities = np.dot(self.hotels_embeddings, pref_embedding.T).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]
        return [
            {**self.hotels_db[i], 'similarity_score': float(similarities[i])}
            for i in top_indices
        ]


class ItineraryPlannerAgent:
    """
    An agent that creates travel itineraries by generating descriptive text using GPT-2.

    Attributes:
        planner (pipeline): Text-generation pipeline from Hugging Face transformers.
    """
    def __init__(self):
        # Uses a language model for generating itineraries
        self.planner = pipeline(
            "text-generation",
            model="gpt2",  # In production, use a more powerful model
            max_length=500,
            truncation=True,
            pad_token_id=50256
        )

    def create_itinerary(self, destination_info: Dict, weather_info: Dict,
                         hotel_info: Dict, duration: int) -> Dict:
        prompt = self._create_prompt(destination_info, weather_info, hotel_info, duration)

        # Generate the itinerary
        response = self.planner(prompt)[0]['generated_text']

        return {
            'itinerary': response,
            'duration': duration,
            'destination': destination_info['name']
        }

    def _create_prompt(self, destination_info: Dict, weather_info: Dict,
                       hotel_info: Dict, duration: int) -> str:
        return f"""Create a {duration}-day itinerary for {destination_info['name']}.
        Weather: {weather_info['best_months'][0]['month']} is the best month.
        Hotel: Staying at {hotel_info[0]['name']}.
        Attractions: {', '.join(destination_info['attractions'])}."""


class SummaryAgent:
    """
    An agent that generates summarized travel plans in the form of personalized emails.

    Attributes:
        llm (pipeline): Text-generation pipeline from Hugging Face transformers for email creation.
    """
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model="gpt2",
            max_length=1000,
            truncation=True,
            pad_token_id=50256
        )

    def calculate_total_price(self, hotel_info: Dict, duration: int) -> float:
        # Calculate total trip price
        hotel_cost = hotel_info[0]['price'] * duration
        # Estimate additional costs (activities, meals, transport)
        daily_expenses = 100  # Simplified example
        additional_costs = daily_expenses * duration

        return hotel_cost + additional_costs

    def create_email(self, trip_data: Dict, client_name: str) -> Dict:
        total_price = self.calculate_total_price(
            trip_data['recommended_hotels'],
            trip_data['itinerary']['duration']
        )

        prompt = f"""
        Dear {client_name},

        Based on your preferences, I'm pleased to present your travel plan:

        Destination: {trip_data['itinerary']['destination']}
        Duration: {trip_data['itinerary']['duration']} days
        Best time to visit: Month {trip_data['weather_analysis']['best_months'][0]['month']}

        Recommended Hotel: {trip_data['recommended_hotels'][0]['name']}

        Itinerary Overview:
        {trip_data['itinerary']['itinerary']}

        Estimated Total Cost: ${total_price}

        Please let me know if you would like any adjustments.
        """

        # Generate email using LLM
        response = self.llm(prompt)[0]['generated_text']

        return {
            'email_content': response,
            'total_price': total_price,
            'summary_data': {
                'destination': trip_data['itinerary']['destination'],
                'duration': trip_data['itinerary']['duration'],
                'hotel': trip_data['recommended_hotels'][0]['name'],
                'best_month': trip_data['weather_analysis']['best_months'][0]['month']
            }
        }


class TravelPlanningSystem:
    def __init__(self):
        self.weather_agent = WeatherAnalysisAgent()
        self.hotel_agent = HotelRecommenderAgent()
        self.itinerary_agent = ItineraryPlannerAgent()
        self.summary_agent = SummaryAgent()

    def setup(self, historical_weather_data: Dict, hotels_database: List[Dict]):
        # Initialize and train the models
        self.weather_agent.train(historical_weather_data)
        self.hotel_agent.add_hotels(hotels_database)

    def plan_trip(self, destination: Dict, preferences: str, duration: int, client_name: str) -> Dict:
        # 1. Weather analysis and best time prediction
        weather_analysis = self.weather_agent.predict_best_time(destination)

        # 2. Hotel search
        recommended_hotels = self.hotel_agent.find_hotels(preferences)

        # 3. Itinerary creation
        itinerary = self.itinerary_agent.create_itinerary(
            destination,
            weather_analysis,
            recommended_hotels,
            duration
        )

        # 4. Create summary email and calculate price
        trip_data = {
            'weather_analysis': weather_analysis,
            'recommended_hotels': recommended_hotels,
            'itinerary': itinerary
        }

        summary = self.summary_agent.create_email(trip_data, client_name)

        return {
            **trip_data,
            'summary': summary
        }


def main():
    # Example data with a full year of weather information
    historical_weather_data = [
        {'month': 1, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.5},
        {'month': 2, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.6},
        {'month': 3, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.7},
        {'month': 4, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.8},
        {'month': 5, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.85},
        {'month': 6, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.9},
        {'month': 7, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.95},
        {'month': 8, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.9},
        {'month': 9, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.85},
        {'month': 10, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.7},
        {'month': 11, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.6},
        {'month': 12, 'latitude': 41.9028, 'longitude': 12.4964, 'weather_score': 0.5},
    ]

    # Sample hotel database
    hotels_database = [
        {
            'name': 'Grand Hotel',
            'description': 'Luxury hotel in city center with spa and restaurant',
            'price': 300
        },
        {
            'name': 'Boutique Resort',
            'description': 'Intimate boutique hotel with personalized service',
            'price': 250
        },
        {
            'name': 'City View Hotel',
            'description': 'Modern hotel with panoramic city views',
            'price': 200
        }
    ]

    # Initialize the system
    system = TravelPlanningSystem()
    system.setup(historical_weather_data, hotels_database)

    # Plan a trip
    destination = {
        'name': 'Rome',
        'latitude': 41.9028,
        'longitude': 12.4964,
        'attractions': ['Colosseum', 'Vatican', 'Trevi Fountain']
    }

    preferences = """Looking for a luxury hotel in the city center,
    preferably with spa facilities and fine dining options"""

    client_name = "John Smith"

    # Generate trip plan
    trip_plan = system.plan_trip(destination, preferences, duration=3, client_name=client_name)

    # Print results in a readable format
    print("\nTRAVEL PLANNING RESULTS:")
    print("-" * 50)
    print(f"Client: {client_name}")
    print(f"Destination: {destination['name']}")
    print("\nGenerated Email:")
    print("-" * 20)
    print(trip_plan['summary']['email_content'])
    print("\nEstimated Total Price:")
    print(f"${trip_plan['summary']['total_price']}")


if __name__ == "__main__":
    main()