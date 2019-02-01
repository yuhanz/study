from rasa_core_sdk import Action
from rasa_core_sdk.events import SlotSet

import requests

class ActionGetGreatPrice(Action):
   def name(self):
      # type: () -> Text
      return "action_get_great_price"

   def run(self, dispatcher, tracker, domain):
      # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict[Text, Any]]
      print('===entities:', tracker.latest_message)

      model = tracker.get_slot('model')
      url = 'https://qa-11-www.edmunds.com/api/inventory/v5/find/?model={}&fields=prices,thirdPartyInfo'.format(model)
      resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'})
      result = resp.json()
      greatPrice = result.get('results')[0].get('thirdPartyInfo').get('priceValidation').get('maxGreatPrice')
      return [SlotSet("greatPrice", '{}'.format(greatPrice))]
