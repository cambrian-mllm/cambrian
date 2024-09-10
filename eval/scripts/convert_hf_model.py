import os

import torch

from cambrian.model import CambrianLlamaForCausalLM


def main(
        ckpt_path: str,
        llm_model_name="lmsys/vicuna-7b-v1.5",
        state_dict_name='full_model.pth',
):

    print(f"""Converting to HF:
    LLM Model: {llm_model_name}
    Checkpoint Path: {ckpt_path}
    State Dict Name: {state_dict_name}
    """)

    state_dict_path = os.path.join(ckpt_path, state_dict_name)
    tpu_state_dict = torch.load(state_dict_path)

    print(f"Loading model from {llm_model_name}")
    from cambrian.model.language_model.cambrian_llama import CambrianConfig
    # from cambrian.model.language_model.cambrian_mistral import CambrianMistralConfig as CambrianConfig
    # from cambrian.model.language_model.cambrian_cohere import CambrianCohereConfig as CambrianConfig
    config = CambrianConfig.from_pretrained(ckpt_path)
    model = CambrianLlamaForCausalLM.from_pretrained(
        llm_model_name,
        config=config,
        cache_dir=None,
        torch_dtype=None,
    )

    # required for Midas unfrozen?
    # vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()

    print(f"Loading state dict from {state_dict_path}")
    model.load_state_dict(tpu_state_dict, strict=True)
    model.generation_config.do_sample = True
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v

    model.save_pretrained(ckpt_path, state_dict=state_dict)
    print(f"Saved converted HF model to {ckpt_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--llm_model_name', type=str,
                        default="lmsys/vicuna-7b-v1.5")
    parser.add_argument('--state_dict_name', type=str,
                        default='full_model.pth')
    args = parser.parse_args()

    main(args.ckpt_path, args.llm_model_name, args.state_dict_name)
