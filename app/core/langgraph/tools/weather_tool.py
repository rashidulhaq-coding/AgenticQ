"""Dummy weather tool for the QA agent.

Returns simulated weather data for any city. This is a stand-in for a real
weather API and demonstrates tool schema with structured I/O.
"""

import json
import random
from typing import Optional

from langchain_core.tools import tool

from app.core.logging import logger


WEATHER_CONDITIONS = [
    "Clear sky",
    "Partly cloudy",
    "Overcast",
    "Light rain",
    "Moderate rain",
    "Thunderstorm",
    "Snow",
    "Fog",
    "Haze",
    "Windy",
]


@tool
def get_weather(city: str, unit: Optional[str] = "celsius") -> str:
    """Get the current weather for a given city.

    Use this tool when the user asks about weather conditions, temperature,
    forecasts, or anything weather-related for a specific location.

    Args:
        city: The name of the city to get weather for.
        unit: Temperature unit, either "celsius" or "fahrenheit". Defaults to "celsius".

    Returns:
        A JSON string containing weather data for the city, including
        temperature, condition, humidity, and wind speed.
    """
    logger.info(
        "tool_input",
        event_type="tool_input",
        tool_name="get_weather",
        args={"city": city, "unit": unit},
    )

    random.seed(hash(city.lower()))

    base_temp_c = random.randint(-5, 40)
    condition = random.choice(WEATHER_CONDITIONS)
    humidity = random.randint(20, 95)
    wind_speed = random.randint(0, 60)
    feels_like_offset = random.randint(-3, 3)

    if unit and unit.lower() == "fahrenheit":
        temp = round(base_temp_c * 9 / 5 + 32, 1)
        feels_like = round((base_temp_c + feels_like_offset) * 9 / 5 + 32, 1)
        temp_unit = "°F"
    else:
        temp = base_temp_c
        feels_like = base_temp_c + feels_like_offset
        temp_unit = "°C"

    result = {
        "city": city,
        "temperature": f"{temp}{temp_unit}",
        "feels_like": f"{feels_like}{temp_unit}",
        "condition": condition,
        "humidity": f"{humidity}%",
        "wind_speed": f"{wind_speed} km/h",
        "unit": unit or "celsius",
        "source": {
            "name": "WeatherDemo",
            "url": "https://weatherdemo.example.com",
        },
    }

    logger.info(
        "tool_output",
        event_type="tool_output",
        tool_name="get_weather",
        result=json.loads(json.dumps(result)),
    )

    return json.dumps(result)