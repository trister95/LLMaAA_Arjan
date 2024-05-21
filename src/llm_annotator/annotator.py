import ujson as json
import os
from ujson import JSONDecodeError
from func_timeout import func_set_timeout
from openai import RateLimitError
from langchain_core import prompts, output_parsers
from langchain_openai import ChatOpenAI



class Annotator:
    def __init__(self, engine: str = 'gpt-3.5-turbo', config_name: str = 'default', dataset: str = None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, 'configs', f'{config_name}.json')
        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        self.dataset = dataset or config['dataset']
        self.task = config['task']
        self.description = config['description']
        self.guidance = config['guidance']
        self.input_format = config['input_format']
        self.output_format = config['output_format']
        self.struct_format = config['struct_format']

        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=engine)

        # Setup prompt and output parsers

        self.prompt_template = prompts.ChatPromptTemplate.from_messages([
            ("system", self.description.replace("{", "{{").replace("}", "}}")),  # Escaping the braces
            ("system", self.guidance),
            ("user", "{input}")
        ])

        self.output_parser = output_parsers.StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser

    def generate_prompt(self, sample, demo = None):
        to_annotate = self.input_format.format(sample['text'])
        if demo:
            demo_annotations = "\n".join(f"{self.input_format.format(d['text'])}\n{self.output_format.format(d['labels'])}" for d in demo)
            return f"here are some examples:\n{demo_annotations}\n \n Please now annotate the following input: \n {to_annotate}"
        else:
            return f"please annote the following input: \n{to_annotate}"
        

    
    @func_set_timeout(60)
    def online_annotate(self, sample, demo=None):
        annotation_prompt = self.generate_prompt(sample, demo)
        retry_count = 0  # Initialize retry counter

        while retry_count < 3:  # Allow up to 3 attempts (initial + 2 retries)
            try:
                response = self.chain.invoke({"input": annotation_prompt})
                parsed_result = json.loads(response)
                return self.postprocess(parsed_result)

            except RateLimitError:
                print("Rate limit exceeded. Please wait and try again.")
                print(f"Problem was with: {annotation_prompt}")
                return None

            except Exception as e:
                print(f"Error during annotation: {e}")
                print(f"Problem was with: {annotation_prompt}")
                retry_count += 1  # Increment retry counter

                if retry_count == 3:
                    print("Max retries reached. Aborting operation.")
                    return None

                print("Retrying...")

        return None
    

    def postprocess(self, result):
        if self.task == 'ner':
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dir_path = os.path.dirname(os.path.dirname(dir_path))
            meta_path = os.path.join(dir_path, f'data/{self.dataset}/meta.json')
            with open(meta_path, 'r') as file:
                meta = json.load(file)
            tagset = meta['tagset']
            outputs = []
            for entity in result:
                if not isinstance(entity, dict):
                    continue
                if 'type' not in entity or 'span' not in entity:
                    continue
                if entity['type'] in tagset:
                    outputs.append(entity)
            return outputs
    
if __name__ == '__main__':
    annotator = Annotator(engine='gpt-3.5-turbo', config_name='en_conll03')
    sample = {"tokens":["因","盛","产","绿","竹","笋","，","被","誉","为","「","绿","竹","笋","的","故","乡","」","的","八","里","，","就","像","台","湾","许","多","大","大","小","小","遥","远","的","乡","镇","，","在","期","待","与","失","落","中","，","承","载","着","生","活","必","需","的","悲","苦","与","欢","乐","，","并","由","于","位","处","边","陲","，","担","负","着","众","人","不","愿","承","受","之","重","。"],"tags":["O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","B-GPE","E-GPE","O","O","O","B-GPE","E-GPE","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O","O"],"text":"因盛产绿竹笋，被誉为「绿竹笋的故乡」的八里，就像台湾许多大大小小遥远的乡镇，在期待与失落中，承载着生活必需的悲苦与欢乐，并由于位处边陲，担负着众人不愿承受之重。","labels":[{"span":"台湾","type":"GPE"},{"span":"八里","type":"GPE"}],"id":"23549"}
    demo = [sample]
    print(annotator.online_annotate(sample))