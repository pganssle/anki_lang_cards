from typing import Final

SENTENCE_PROMPT: Final[
    str
] = """\
You are an expert Spanish teacher and lexicographer, whose specialty is designing excellent example sentences that show off the range of uses of a given word.

I am going to give you valid JSON representing a term ("term") in Spanish and its meaning in English ("meaning"). I would like you to generate valid JSON containing 4 example sentences in idiomatic Spanish that use the term accompanied by a translation from the original Spanish into English; the example sentences should use the sense of the term that has the meaning specified by the English meaning. When the word is a verb, select different moods and tenses in the different sentences. The translations do not need to be word-for-word translations, it is more important that they capture the general spirit of the original Spanish.

In some of your example sentences, it would be a bonus if they contain useful or interesting facts.

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
            "spanish": "En la Edad Media, las casas se construían a menudo con bahareque, que consistía en revestir ramas y paja con arcilla, barro y estiércol.",
            "english": "In the Middle Ages, houses were often made of wattle and daub, which involved plastering branches and straw with clay, mud and dung."
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
            "spanish": "Los delfines pueden «ver» rebotando ondas sonoras en su entorno."
            "english": "Dolphins can 'see' by bouncing sound waves off their surroundings."
        },
        {
            "spanish": "¡Rebota más alto!",
            "english": "Bounce higher!"
        },
        {
            "spanish": "Rebotaremos la luz en la luna.",
            "english": "We will bounce the light off the moon."
        },
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

In some of your example sentences, it would be a bonus if they contain useful or interesting facts.

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
                "spanish": "En la Edad Media, las casas se construían a menudo con bahareque, que consistía en revestir ramas y paja con arcilla, barro y estiércol.",
                "english": "In the Middle Ages, houses were often made of wattle and daub, which involved plastering branches and straw with clay, mud and dung."
            },
            {
                "spanish": "Voy a limpiar mi casa este fin de semana.",
                "english": "I am going to clean my house this weekend."
            },
            {
                "spanish": "Me gusta pasar tiempo en casa de mis padres.",
                "english": "I like to spend time at my parents' house."
            },
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
                "spanish": "Los delfines pueden «ver» rebotando ondas sonoras en su entorno."
                "english": "Dolphins can 'see' by bouncing sound waves off their surroundings."
            },
            {
                "spanish": "Rebotaremos la luz en la luna.",
                "english": "We will bounce the light off the moon."
            },
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

ORG_TO_CSV_PROMPT_TEMPLATE: Final[
    str
] = """\
Task:

Translate notes on Spanish words into well-formatted CSVs with the definition, part of speech and (if applicable), article.

Details:

I take notes in org mode on Spanish words I encounter. Sometimes the note is a word in English or Spanish, sometimes it is spanish and the english translation I am expecting.

- If the word is a valid Spanish word and there is no English definition, please use the most common English definition, or if more than one definition is very common, include as many as 3.
- If the word is not a valid Spanish word and there is no English definition, please try to choose a Spanish word that is most likely to sound like the word I've written; prefer common words over uncommon ones.
- If the word is not a valid Spanish word and an English definition is present, try to choose a Spanish word that sounds like the Spanish word I've included and has a definition matching the English word.
- If there is a Spanish word with a definition and the definition of the Spanish word is not correct, please correct the definition, but include an asterisk followed by the original, erroneous English defintiion in parentheses.

Output format:

The output format should be a CSV with the following header:

word,meaning,part_of_speech,article

Only populate "article" for nouns. Valid parts of speech are:

nm = masculine noun
nf = feminine noun
nm/f = masculine/feminine noun (varies depending on the gender of the subject)
vtr = transitive verb
vi = intransitive verb
v phr = verbal phrase
adv = adverb
adj = adjective
adj mf = adjective using one form for both masculine and feminine

Output the words in the same order that they came in. Do not skip any rows. If there is more than one Spanish word on a row, split it into multiple definitions. If there is a slight difference between the words on a given word, try to add a *terse* note about what the difference is in the English defintion, e.g.:

gemelo / mellizo = twin

Would turn into:

gemelo/a,"twin (identical)",nm/f,el/la
mellizo/a,"twin, fraternal twin",nm/f,el/la

If a word is only commonly used in Latin America, append (LatAm) after the English definition. If it is only commonly used in Spain, append (Spain) after the English definition.

ONLY OUTPUT THE WELL-FORMATTED CSV. Do not give any explanations or other extraneous text.

Examples:

---

INPUT:

%(org_text)s

OUTPUT:
"""
