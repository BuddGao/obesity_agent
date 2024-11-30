# health evaluator of the system
from .base_agent import Agent
from utils.register import register_class, registry

@register_class(alias="Agent.Evaluator.GPT")
class Evaluator(Agent):
    def __init__(self, args, evaluator_info=None):
        engine = registry.get_class("Engine.GPT")(
            openai_api_key=args.evaluator_openai_api_key, 
            openai_api_base=args.evaluator_openai_api_base,
            openai_model_name=args.evaluator_openai_model_name, 
            temperature=args.evaluator_temperature, 
            max_tokens=args.evaluator_max_tokens,
            top_p=args.evaluator_top_p,
            frequency_penalty=args.evaluator_frequency_penalty,
            presence_penalty=args.evaluator_presence_penalty
        )
        if evaluator_info is None:
            self.system_message = \
                "You are an evaluator. You are responsible for evaluating the health conditions of any residents.\n" + \
                "Your primary focus is to assess the obesity status of the residents based on their medical records.\n" + \
                "Consider the following criteria for obesity assessment:\n" + \
                "1. Family History: Look for any family history of overweight or obesity.\n" + \
                "2. Dietary Habits: Evaluate the dietary habits and nutritional intake.\n" + \
                "3. Physical Activity: Assess the level of physical activity and sedentary behavior.\n" + \
                "4. Smoking and Alcohol Consumption: Consider the impact of smoking and alcohol consumption.\n"
        else:
            self.system_message = evaluator_info

        super(Evaluator, self).__init__(engine)

    @staticmethod
    def add_parser_args(parser):
        parser.add_argument('--evaluator_openai_api_key', type=str, help='API key for OpenAI')
        parser.add_argument('--evaluator_openai_api_base', type=str, help='API base for OpenAI')
        parser.add_argument('--evaluator_openai_model_name', type=str, help='API model name for OpenAI')
        parser.add_argument('--evaluator_temperature', type=float, default=0.0, help='temperature')
        parser.add_argument('--evaluator_max_tokens', type=int, default=2048, help='max tokens')
        parser.add_argument('--evaluator_top_p', type=float, default=1, help='top p')
        parser.add_argument('--evaluator_frequency_penalty', type=float, default=0, help='frequency penalty')
        parser.add_argument('--evaluator_presence_penalty', type=float, default=0, help='presence penalty')

    def evaluate_health_condition(self, resident_profile, medical_records):
        system_message = self.system_message + "\n" + \
            "Resident Profile:\n" + resident_profile + "\n" + \
            "Medical Records:\n" + medical_records + "\n"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Please evaluate the health condition of the resident based on the provided information."}
        ]
        response = self.engine.get_response(messages)
        return response

    def evaluate_obesity_status(self, record):
        system_message = self.system_message + "\n" + \
            "Resident Record:\n" + str(record) + "\n" 
            
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Please evaluate the resident's obesity status in an integer form from 0 to 100, where 0 means underweight and 100 means overweight, based on the provided record."}
        ]
        response = self.engine.get_response(messages)
        return int(response.strip())

    def should_go_to_doctor(self, record):
        system_message = self.system_message + "\n" + \
            "Resident Record:\n" + str(record) + "\n" 
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Please evaluate whether the resident should go to the doctor, based on the provided record."}
        ]
        response = self.engine.get_response(messages)
        return response.strip().lower() == 'yes'

    def predict_future_status(self, record):
        system_message = self.system_message + "\n" + \
            "Resident Record:\n" + str(record) + "\n" 
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "Please evaluate if the resident's condition is likely to get worse or better in the next round of obesity, based on the provided record."}
        ]
        response = self.engine.get_response(messages)
        return response.strip()

    def speak(self, resident_profile, medical_records):
        evaluation = self.evaluate_health_condition(resident_profile, medical_records)
        obesity_score = self.evaluate_obesity_status(resident_profile)
        doctor_recommendation = self.should_go_to_doctor(resident_profile)
        future_status = self.predict_future_status(resident_profile)
        return {
            "evaluation": evaluation,
            "obesity_score": obesity_score,
            "should_go_to_doctor": doctor_recommendation,
            "future_status": future_status
        }