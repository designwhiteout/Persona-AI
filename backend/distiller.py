"""
Character distillation: extracts personality traits, speech patterns,
and generates a system prompt from sample texts using OpenAI.
"""
import json
from openai import AsyncOpenAI

from models import CharacterProfile, CharacterTrait, SpeechPattern
from config import settings

_DISTILL_SYSTEM = """You are an expert character analyst and writer.
Your job is to deeply analyze text samples and extract the essential essence of a character.
Always respond with valid JSON matching the requested schema exactly."""

_DISTILL_PROMPT = """Analyze the following character information and sample texts, then extract the character's essence.

Character Name: {name}
Character Description: {description}
Background: {background}
Additional Context: {additional_context}

Sample Texts:
{samples}

Return a JSON object with this exact structure:
{{
  "traits": [
    {{
      "name": "trait name",
      "description": "detailed description of this trait as expressed by the character",
      "intensity": 0.0-1.0
    }}
  ],
  "speech_pattern": {{
    "vocabulary_style": "description of word choices, formality level, vocabulary richness",
    "sentence_structure": "description of how the character constructs sentences",
    "tone": "emotional tone and attitude in communication",
    "quirks": ["specific recurring phrases", "speech habits", "unique expressions"]
  }},
  "system_prompt": "A detailed system prompt (200-400 words) that will make an AI roleplay as this character. Include personality, speech style, values, how they respond to different situations, and what makes them unique. Write in second person (You are...)."
}}

Be specific and detailed. Extract at least 5 personality traits. The system prompt should fully capture the character's voice."""


async def distill_character(
    character: CharacterProfile,
    additional_context: str = "",
) -> tuple[list[CharacterTrait], SpeechPattern, str]:
    """
    Returns (traits, speech_pattern, system_prompt) extracted from character data.
    """
    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )

    samples_text = "\n\n---\n\n".join(
        f"[Sample {i+1}]\n{text}"
        for i, text in enumerate(character.sample_texts)
    ) if character.sample_texts else "(No sample texts provided — infer from description and background)"

    prompt = _DISTILL_PROMPT.format(
        name=character.name,
        description=character.description,
        background=character.background or "Not specified",
        additional_context=additional_context or "None",
        samples=samples_text,
    )

    response = await client.chat.completions.create(
        model=settings.distill_model,
        messages=[
            {"role": "system", "content": _DISTILL_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = json.loads(response.choices[0].message.content)

    traits = [CharacterTrait(**t) for t in raw.get("traits", [])]
    speech_pattern = SpeechPattern(**raw["speech_pattern"])
    system_prompt = raw["system_prompt"]

    return traits, speech_pattern, system_prompt
