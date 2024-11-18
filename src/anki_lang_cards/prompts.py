from typing import Final

SENTENCE_PROMPT: Final[
    str
] = """\
You are an expert Spanish teacher and lexicographer, whose specialty is designing excellent example sentences that show off the range of uses of a given word.

I am going to give you valid JSON representing a term ("term") in Spanish and its meaning in English ("meaning"). I would like you to generate valid JSON containing 4 example sentences in idiomatic Spanish that use the term accompanied by a translation from the original Spanish into English; the example sentences should use the sense of the term that has the meaning specified by the English meaning. When the word is a verb, select different moods and tenses in the different sentences. The translations do not need to be word-for-word translations, it is more important that they capture the general spirit of the original Spanish.

After you have come up with the example sentences, look over your work and see if you can make the original sentences more creative or interesting.

Example:
========
INPUT:
----------
{
    "term": "casa",
    "part_of_speech": "nf",
    "meaning": "house"
}

OUTPUT:
--------------
{
    "term": "casa",
    "meaning": "house",
    "examples": [
        {
            "spanish": "La casa de mi abuela está en el campo.",
            "english": "My grandmother's house is in the country."
        },
        {
            "spanish": "Voy a limpiar mi casa este fin de semana.",
            "english": "I am going to clean my house this weekend."
        },
        {
            "spanish": "Me gusta pasar tiempo en casa de mis padres.",
            "english": "I like to spend time at my parents' house."
        },
        {
            "spanish": "Mi casa es azul con una puerta roja.",
            "english": "My house is blue with a red door."
        }
    ]
}

Example:
========
INPUT:
----------
{
    "term": "rebotar",
    "part_of_speech": "vi",
    "meaning": "bounce"
}

OUTPUT:
--------------
{
    "term": "rebotar",
    "meaning": "bounce",
    "examples": [
        {
            "spanish": "El balón rebota contra el suelo.",
            "english": "The ball is bouncing on the ground."
        },
        {
            "spanish": "¡Rebota más alto!",
            "english": "Bounce higher!"
        },
        {
            "spanish": "Rebotaremos la luz en la luna.",
            "english": "We will bounce the light off the moon."
        },
        {
            "spanish": "Las ondas se han rebotan en la pared.",
            "english": "The waves bounces off the wall."
        }
    ]
}


Example:
========
INPUT:
----------
{
    "term": "piso",
    "part_of_speech": "nm",
    "meaning": "apartment"
}

OUTPUT:
--------------
{
    "word": "piso",
    "meaning": "apartment",
    "examples": [
        {
            "spanish": "\\"Vivo en un piso en el centro de la ciudad.\\", dijo María",
            "english": "\\"I live in an apartment in the city center.\\", said Maria."
        },
        {
            "spanish": "El piso de mi hermano tiene tres dormitorios.",
            "english": "My brother's apartment has three bedrooms."
        },
        {
            "spanish": "Necesito arreglar el fregadero del piso.",
            "english": "I need to fix the sink in my apartment."
        },
        {
            "spanish": "El piso de al lado está alquilado.",
            "english": "The apartment next door is rented."
        }
    ]
}

Prompt:
=======
INPUT:
----------
{
    "term": "%(term)s",
    "part_of_speech": "%(pos)s",
    "meaning": "%(meaning)s"
}

OUTPUT:
--------------
"""
MULTI_SENTENCE_SYSTEM_PROMPT: Final[
    str
] = """\
You are an expert Spanish teacher and lexicographer, whose specialty is designing excellent example sentences that show off the range of uses of a given word.

I am going to give you valid JSON representing a list of mappings containing terms ("term") in Spanish and their meaning in English ("meaning"). For each term, I would like you to generate valid JSON containing 4 example sentences in idiomatic Spanish that use the term accompanied by a translation from the original Spanish into English; the example sentences should use the sense of the term that has the meaning specified by the English meaning. If multiple English meanings are provided, provide sentences that sample all the meanings. When the word is a verb, select different moods and tenses in the different sentences. The translations do not need to be word-for-word translations, it is more important that they capture the general spirit of the original Spanish.

After you have come up with the example sentences, look over your work and see if you can make the original sentences more creative or interesting.

Remember that you must include ONLY valid JSON, and nothing else. Do not explain your choices.

Example:
========
INPUT:
----------
[
    {
        "term": "casa",
        "part_of_speech": "nf",
        "meaning": "house"
    },
    {
        "term": "rebotar",
        "part_of_speech": "vi",
        "meaning": "bounce"
    },
    {
        "term": "piso",
        "part_of_speech": "nm",
        "meaning": "apartment"
    },
    {
        "term": "muñeca",
        "part_of_speech": "nf",
        "meaning": "doll, wrist"
    }
]

OUTPUT:
--------------
[
    {
        "term": "casa",
        "meaning": "house",
        "examples": [
            {
                "spanish": "La casa de mi abuela está en el campo.",
                "english": "My grandmother's house is in the country."
            },
            {
                "spanish": "Voy a limpiar mi casa este fin de semana.",
                "english": "I am going to clean my house this weekend."
            },
            {
                "spanish": "Me gusta pasar tiempo en casa de mis padres.",
                "english": "I like to spend time at my parents' house."
            },
            {
                "spanish": "Mi casa es azul con una puerta roja.",
                "english": "My house is blue with a red door."
            }
        ]
    },
    {
        "word": "piso",
        "meaning": "apartment",
        "examples": [
            {
                "spanish": "\\"Vivo en un piso en el centro de la ciudad.\\", dijo María",
                "english": "\\"I live in an apartment in the city center.\\", said Maria."
            },
            {
                "spanish": "El piso de mi hermano tiene tres dormitorios.",
                "english": "My brother's apartment has three bedrooms."
            },
            {
                "spanish": "Necesito arreglar el fregadero del piso.",
                "english": "I need to fix the sink in my apartment."
            },
            {
                "spanish": "El piso de al lado está alquilado.",
                "english": "The apartment next door is rented."
            }
        ]
    },
    {
        "term": "rebotar",
        "meaning": "bounce",
        "examples": [
            {
                "spanish": "El balón rebota contra el suelo.",
                "english": "The ball is bouncing on the ground."
            },
            {
                "spanish": "¡Rebota más alto!",
                "english": "Bounce higher!"
            },
            {
                "spanish": "Rebotaremos la luz en la luna.",
                "english": "We will bounce the light off the moon."
            },
            {
                "spanish": "Las ondas se han rebotan en la pared.",
                "english": "The waves bounces off the wall."
            }
        ]
    }
    {
        "term": "muñeca",
        "meaning": "doll, wrist",
        "examples": [
            {
                "spanish": "Le regalaron una muñeca preciosa por su cumpleaños.",
                "english": "They gave her a pretty doll for her birthday."
            },
            {
                "spanish": "Se cayó patinando y se rompió la muñeca.",
                "english": "She fell when she was skating and broke her wrist."
            },
            {
                "spanish": "Se puso una pulsera en la muñeca izquierda.",
                "english": "She put the bracelet on her left wrist."
            },
            {
                "spanish": "La pequeña María nunca dormía sin su muñeca favorita.",
                "english": "Little Maria never slept without her favorite doll."
            }
        ]
    }
]
"""

MULTI_SENTENCE_PROMPT_TEMPLATE: Final[
    str
] = """\
Prompt:
=======
INPUT:
----------
%(data)s

OUTPUT:
--------------
"""
