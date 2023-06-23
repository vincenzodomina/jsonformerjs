import { PreTrainedModel, PreTrainedTokenizer, LogitsWarper, StoppingCriteria } from '@xenova/transformers';
import nj from 'numjs';

type JsonSchema = {
  type: string;
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
};

type JsonValue = number | boolean | string | any | any[];
type JsonObject = Record<string, JsonValue>;
type JsonArray = JsonValue[];

export class Jsonformer {
  private model: any; // PreTrainedModel;
  private tokenizer: any; // PreTrainedTokenizer;
  private json_schema: JsonSchema;
  private prompt: string;
  private generation_marker: string;
  private debug_on: boolean;
  private max_array_length: number;
  private max_number_tokens: number;
  private temperature: number;
  private max_string_token_length: number;
  private value: JsonObject = {};
  //private number_logit_processor: OutputNumbersTokens;

  constructor(
    model: any, //PreTrainedModel,
    tokenizer: any, //PreTrainedTokenizer,
    json_schema: JsonSchema,
    prompt: string,
    debug = false,
    max_array_length = 10,
    max_number_tokens = 6,
    temperature = 1.0,
    max_string_token_length = 10
  ) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.json_schema = json_schema;
    this.prompt = prompt;

    //this.number_logit_processor = new OutputNumbersTokens(this.tokenizer, this.prompt);

    this.generation_marker = "|GENERATION|";

    this.debug_on = debug;
    this.max_array_length = max_array_length;

    this.max_number_tokens = max_number_tokens;
    this.temperature = temperature;
    this.max_string_token_length = max_string_token_length;
  }

  private debug(caller: string, value: string, is_prompt: boolean = false): void {
    if (this.debug_on) {
      if (is_prompt) {
        console.log('\x1b[32m' + caller); // Print caller in green
        console.log('\x1b[33m' + value); // Print value in yellow
      } else {
        console.log('\x1b[32m' + caller); // Print caller in green
        console.log('\x1b[34m' + value); // Print value in blue
      }
    }
  }

  private async generate_number(temperature: number | null = null, iterations = 0): Promise<number> {
    const prompt = this.get_prompt();
    this.debug('[generate_number]', prompt, true);
    const input_tokens = this.tokenizer.encode(prompt, { return_tensors: 'pt' }).to(this.model.device);
    const response = await this.model.generate(input_tokens, {
      max_new_tokens: this.max_number_tokens,
      num_return_sequences: 1,
      //logits_processor: [this.number_logit_processor],
      //stopping_criteria: [new NumberStoppingCriteria(this.tokenizer, input_tokens[0].length)],
      temperature: temperature ?? this.temperature,
      pad_token_id: this.tokenizer.eos_token_id,
    });
    const decoded_response = this.tokenizer.decode(response[0], { skip_special_tokens: true });

    const result = decoded_response.slice(prompt.length).trim().replace('.', '');
    this.debug('[generate_number]', result);
    try {
      return parseFloat(result);
    } catch (error) {
      if (iterations > 3) {
        throw new Error('Failed to generate a valid number');
      }
      return this.generate_number(this.temperature * 1.3);
    }
  }

  private async generate_boolean(): Promise<boolean> {
    const prompt = this.get_prompt();
    this.debug('[generate_boolean]', prompt, true);

    const input_tensor = this.tokenizer.encode(prompt, { return_tensors: 'pt' });
    const output = this.model.forward(input_tensor.to(this.model.device));
    const logits = output.logits[0, -1];

    const true_token_id = this.tokenizer.convert_tokens_to_ids('true');
    const false_token_id = this.tokenizer.convert_tokens_to_ids('false');

    const result = logits[true_token_id] > logits[false_token_id];

    this.debug('[generate_boolean]', result.toString());

    return result;
  }

  private async generate_string(): Promise<string> {
    const prompt = this.get_prompt() + '"';
    this.debug('[generate_string]', prompt, true);
    const input_tokens = this.tokenizer.encode(prompt, { return_tensors: 'pt' }).to(this.model.device);

    const response = await this.model.generate(input_tokens, {
      max_new_tokens: this.max_string_token_length,
      num_return_sequences: 1,
      temperature: this.temperature,
      //stopping_criteria: [new StringStoppingCriteria(this.tokenizer, input_tokens[0].length)],
      pad_token_id: this.tokenizer.eos_token_id,
    });

    let decoded_response = this.tokenizer.decode(response[0], { skip_special_tokens: true });

    this.debug('[generate_string]', '|' + decoded_response + '|');

    if (decoded_response.count('"') < 1) {
      return decoded_response;
    }

    return decoded_response.split('"')[0].trim();
  }

  private async generate_object(properties: Record<string, any>, obj: Record<string, any>): Promise<Record<string, any>> {
    for (const key in properties) {
      const schema = properties[key];
      this.debug('[generate_object] generating value for', key);
      obj[key] = await this.generate_value(schema, obj, key);
    }
    return obj;
  }

  private async generate_value(schema: Record<string, any>, obj: Record<string, any> | any[], key: string | null = null): Promise<any> {
    const schema_type = schema['type'];
    if (schema_type === 'number') {
      if (key) {
        (obj as Record<string, any>)[key] = this.generation_marker;
      } else {
        (obj as any[]).push(this.generation_marker);
      }
      return this.generate_number();
    } else if (schema_type === 'boolean') {
      if (key) {
        (obj as Record<string, any>)[key] = this.generation_marker;
      } else {
        (obj as any[]).push(this.generation_marker);
      }
      return this.generate_boolean();
    } else if (schema_type === 'string') {
      if (key) {
        (obj as Record<string, any>)[key] = this.generation_marker;
      } else {
        (obj as any[]).push(this.generation_marker);
      }
      return this.generate_string();
    } else if (schema_type === 'array') {
      const newArray: any[] = [];
      (obj as Record<string, any>)[key!] = newArray;
      return this.generate_array(schema['items'], newArray);
    } else if (schema_type === 'object') {
      const newObj = {};
      if (key) {
        (obj as Record<string, any>)[key] = newObj;
      } else {
        (obj as any[]).push(newObj);
      }
      return this.generate_object(schema['properties'], newObj);
    } else {
      throw new Error(`Unsupported schema type: ${schema_type}`);
    }
  }

  private async generate_array(item_schema: Record<string, any>, obj: any[]): Promise<any[]> {
    for (let i = 0; i < this.max_array_length; i++) {
      const element = await this.generate_value(item_schema, obj);
      obj[-1] = element;

      obj.push(this.generation_marker);
      const input_prompt = this.get_prompt();
      obj.pop();
      const input_tensor = this.tokenizer.encode(input_prompt, { return_tensors: 'pt' });
      const output = this.model.forward(input_tensor.to(this.model.device));
      const logits = output.logits[0, -1];

      const top_indices = logits.topk(30).indices;
      const sorted_token_ids = top_indices[logits[top_indices].argsort({ descending: true })];

      let found_comma = false;
      let found_close_bracket = false;

      for (const token_id of sorted_token_ids) {
        const decoded_token = this.tokenizer.decode(token_id);
        if (decoded_token.includes(',')) {
          found_comma = true;
          break;
        }
        if (decoded_token.includes(']')) {
          found_close_bracket = true;
          break;
        }
      }

      if (found_close_bracket || !found_comma) {
        break;
      }
    }
    return obj;
  }

  private get_prompt(): string {
    const template = `{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}`;
    let progress = JSON.stringify(this.value);
    const gen_marker_index = progress.indexOf(`"${this.generation_marker}"`);
    if (gen_marker_index !== -1) {
      progress = progress.slice(0, gen_marker_index);
    } else {
      throw new Error('Failed to find generation marker');
    }

    const prompt = template.replace('{prompt}', this.prompt)
      .replace('{schema}', JSON.stringify(this.json_schema))
      .replace('{progress}', progress);

    return prompt;
  }

  public async call(): Promise<Record<string, any>> {
    this.value = {};
    const properties = this.json_schema['properties'];
    if (!properties) {
      throw new Error('Missing properties in JSON schema');
    }
    const generated_data = await this.generate_object(properties, this.value);
    return generated_data;
  }
}


class StringStoppingCriteria extends StoppingCriteria {
  tokenizer: PreTrainedTokenizer;
  promptLength: number;

  constructor(tokenizer: PreTrainedTokenizer, promptLength: number) {
    super();
    this.tokenizer = tokenizer;
    this.promptLength = promptLength;
  }

  call(inputIds: nj.NdArray): boolean {
    if (inputIds.shape[1] <= this.promptLength) {
      return false;
    }

    const lastTokenId = inputIds.get(0, inputIds.shape[1] - 1);
    const lastToken = this.tokenizer.decode(lastTokenId, { skipSpecialTokens: true });

    const result = lastToken.includes('"');

    return result;
  }
}


class NumberStoppingCriteria extends StoppingCriteria {
  tokenizer: PreTrainedTokenizer;
  precision: number;
  promptLength: number;

  constructor(tokenizer: PreTrainedTokenizer, promptLength: number, precision: number = 3) {
    super();
    this.tokenizer = tokenizer;
    this.precision = precision;
    this.promptLength = promptLength;
  }

  call(inputIds: nj.NdArray): boolean {
    const decoded = this.tokenizer.decode(
      inputIds.slice([0, this.promptLength]),
      { skipSpecialTokens: true }
    );

    if (decoded.split('.').length - 1 > 1) {
      return true;
    }

    if (
      decoded.split('.').length - 1 === 1 &&
      decoded.trim().split('.')[1].length > this.precision
    ) {
      return true;
    }

    if (
      decoded.length > 1 &&
      decoded.split('').some((c: any) => /\d/.test(c)) &&
      [' ', '\n'].includes(decoded[decoded.length - 1])
    ) {
      return true;
    }

    return false;
  }
}


class OutputNumbersTokens extends LogitsWarper {
  tokenizer: PreTrainedTokenizer;
  tokenizedPrompt: any;
  allowedMask: nj.NdArray;

  constructor(tokenizer: PreTrainedTokenizer, prompt: string) {
    super();
    this.tokenizer = tokenizer;
    this.tokenizedPrompt = tokenizer(prompt, { return_tensors: 'pt' });
    const vocabSize = tokenizer.vocabSize;
    this.allowedMask = nj.zeros([vocabSize]);

    for (const [_, tokenId] of tokenizer.getVocab().entries()) {
      const tokenStr = tokenizer.decode(tokenId).trim();

      if (tokenStr === '' || (
        tokenStr.split('').every((c: any) => /\d|\./.test(c)) &&
        (tokenStr.match(/\./g) || []).length <= 1
      )) {
        this.allowedMask.set(tokenId, 1);
      }
    }
  }

  call(_: any, scores: nj.NdArray): nj.NdArray {
    const mask = nj.tile(this.allowedMask, [scores.shape[0]]);
    const newScores = scores.multiply(mask).add(scores.not().multiply(-Infinity));

    return newScores;
  }
}
