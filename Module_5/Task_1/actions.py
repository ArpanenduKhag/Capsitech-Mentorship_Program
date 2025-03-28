from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionSearchRestaurant(Action):
    def name(self) -> Text:
        return "action_search_restaurant"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        cuisine = tracker.get_slot("cuisine")

        # Dummy restaurant list
        restaurant_list = {
            "Chinese": ["Dragon Palace", "Golden Wok"],
            "Italian": ["Pasta Paradise", "Little Italy"],
        }

        response = f"Here are some {cuisine} restaurants:\n"
        response += "\n".join(restaurant_list.get(cuisine, ["No results found."]))

        dispatcher.utter_message(text=response)
        return []
