from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionSearchRestaurants(Action):

    def name(self) -> Text:
        return "action_show_slots"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        cuisine = tracker.get_slot("cuisine")
        location = tracker.get_slot("location")
        message = f"Current slots:\n- cuisine: {cuisine}\n- location: {location}"
        dispatcher.utter_message(text=message)
        return []