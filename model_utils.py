import os
import torch
import run_train as rt

from javalang.parser import JavaSyntaxError

from scad import mutation as M
from scad.tokenize import func_tokenize

from scad.data import Vocabulary, BPEEncoder
from scad.data import transforms as T
from config import TrainConfig


def _tokenize_java(source_code):
    try:
        return func_tokenize(source_code, 'java')
    except JavaSyntaxError:
        dummy_code = "public class Test { %s }" % source_code
        return func_tokenize(dummy_code, 'java')


def _load_mask_fn(mutation_type):

    if mutation_type == "binary": return M.binary_location_mask
    if mutation_type == "varmisuse": return M.varmisuse_location_mask
    if mutation_type == "funcmisuse": return M.funcmisuse_location_mask

    raise ValueError("Unknown mutation type: %s" % mutation_type)


class InferenceModel:

    def __init__(self, config, model):
        self.config = config
        self.model  = model

    def _tokenize(self, source_code):
        if self.config.lang == "java": return _tokenize_java(source_code)

        return func_tokenize(source_code, self.config.lang)

    def _parse_result(self, tokens, model_output, temp = 1.0):
        
        if isinstance(model_output, tuple):
            loc_logits, repair_logits = model_output
        elif model_output.shape[1] == 2:
            loc_logits, repair_logits = model_output[:, 0, :], model_output[:, 1, :]
        else:
            loc_logits, repair_logits = None, model_output
        
        output = {}

        if loc_logits is not None:
            loc_probs = torch.nn.Softmax(dim = -1)(loc_logits[0]).cpu()
            output["localization"] = [float(t) for t in loc_probs]

        repair_logits /= temp
        repair_probs = torch.nn.Softmax(dim = -1)(repair_logits[0]).cpu()
        repair_result = {}
        
        for i, repair_prob in enumerate(repair_probs):
            if i < len(tokens):
                token = tokens[i]
            else:
                token = self.config.targets[i - len(tokens)]

            if token not in repair_result: repair_result[token] = 0.0
            repair_result[token] += repair_prob.item()
        
        output["repair"] = {k: v for k, v in repair_result.items() if v > 0}
        return output


    def preprocess(self, source_code):
        """Preprocess pipline results in tokens and mask"""
        
        mask_fn = _load_mask_fn(self.config.mutator_type)

        result = {}

        for key, D in self._tokenize(source_code).items():
            mask = mask_fn(D["tokens"], D["types"], lang = self.config.lang)
            result[key] = {"tokens": D["tokens"], "mask": mask}

        return result


    def inference(self, tokens, mask = None, mask_repair = False, temp = 1.0):
        if mask is None: mask = [1] * len(tokens)
        
        instance = {"tokens": tokens, "mask": mask, "location": [], "target": []}

        pipeline = T.SequentialTransform([
            T.BugExampleLoader(self.config.targets)
                 if self.config.targets else T.load_varmisuse_example,
            T.AnnotatedCodeToData(
                T.SubwordEncode(self.config.encoder)
            )
        ])

        data = pipeline(instance).unsqueeze(0).to(self.config.device)

        args = {"tokens": data.input_ids, "token_mask": data.mask}

        if mask_repair:
            mask_locations = [i for i, t in enumerate(tokens) if t == "[M]"]

            assert len(mask_locations) == 1, "We can only repair one location!"
            
            mask_location = mask_locations[0]

            assert mask_location in mask, "Cannot repair mask since it does not belong to a %s type" % self.config.mutator_type

            repair_mask = torch.zeros(data.input_ids.shape[:2])
            repair_mask[0, mask_location] = 1
            repair_mask = repair_mask.to(self.config.device)

            args["repair_mask"] = repair_mask

        with torch.no_grad():
            logits = self.model(**args)

        return self._parse_result(tokens, logits, temp)


    def __call__(self, source_code):
        """Full pipeline from parser over tokens to result"""

        output = {}

        for func_name, D in self.preprocess(source_code).items():
            output[func_name] = self.inference(**D)

        return output


def load_model_from_dir(model_dir, model_type = "binary", lang = "java", arch_type = "loc_repair"):
    
    # Prepare config
    config = TrainConfig("test", model_dir)
    config.max_test_length = 512
    config.sinoid_pos = True
    config.model_type = arch_type

    if model_type.startswith("repair"):
        model_type = model_type.split("_")[1]
        config.model_type = "repair"
    
    #Extra infos
    config.mutator_type = model_type
    config.lang = lang

    target_path = os.path.join(model_dir, "target_vocab.txt")
    if os.path.exists(target_path):
        config.target_path = target_path

    config = rt.init_config(config)

    # Setup Train environment
    enable_gpu = not config.no_cuda and torch.cuda.is_available()
    config.n_gpu = torch.cuda.device_count() if enable_gpu else 0
    config.device = torch.device("cuda" if enable_gpu else "cpu")

    model = rt.setup_model(config)
    model = rt.load_checkpoint(config, model, os.path.join(model_dir, "model.pt"))
    model = model.to(config.device)
    model.eval()

    return InferenceModel(config, model)
