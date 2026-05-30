# =============================================================================
# intent_examples.py — Shapatar Bot v2
# =============================================================================
#
# WHAT THIS FILE IS:
# Training data for the embedding-based intent classifier.
# Each intent has a list of natural example sentences showing the model
# what that intent looks and feels like in real Karachi conversation.
#
# WHAT THIS FILE IS NOT:
# This is not operational data — these strings are never used in bot
# responses. They are used exactly once: when embeddings.py builds the
# intent index at startup. After that, only the vectors matter.
#
# HOW TO IMPROVE CLASSIFICATION:
# Add more examples to any intent that is misclassifying.
# The examples should be natural sentences your friends would actually type.
# Aim for diversity — different vocabulary, different structures,
# different lengths, different mixes of Urdu and English.
#
# MINIMUM: 12 examples per intent
# SWEET SPOT: 20-25 examples per intent
# DIMINISHING RETURNS: beyond 40 examples per intent
#
# LANGUAGE NOTES:
# - Mix Roman Urdu and English
# - Include spelling variants: nahi/nai, yaar/yar, haan/han
# - Include both short and long phrasings
# - Include both explicit and implicit expressions of each intent
# =============================================================================


INTENT_EXAMPLES = {

    # -------------------------------------------------------------------------
    # INVITATION
    # Someone proposing to go somewhere or do something together.
    # Most common intent in casual conversation — needs the most coverage.
    # Implicit invitations ("aaj kuch karte hain") are hardest for rules
    # but embedding handles them naturally.
    # -------------------------------------------------------------------------
    "invitation": [
        # Direct proposals with chalo/chal
        "yaar chal chalte hain bahar",
        "chal yaar nikal chalte hain",
        "bhai chalte hain kahan pe",
        "oye chalein kya aaj",
        "chalo yaar kuch karte hain",

        # Nikal family
        "nikal na yaar",
        "nikal chalte hain raat ko",
        "yaar nikalna chahiye aaj",
        "abey nikal yaar kya kar raha hai ghar mein",

        # Implicit invitations — no obvious trigger word
        "aaj kuch karte hain yaar",
        "bhai aaj plan kya hai",
        "yaar scene banao koi",
        "kuch solid karte hain aaj",
        "bhai bore ho raha hun ghumne chalte hain",

        # Activity-specific invitations
        "yaar khaane chalte hain bahar",
        "bhai movie dekhne chalte hain",
        "chai peene chalte hain yaar",
        "yaar sea view chalte hain raat ko",
        "bhai drive pe nikalte hain",

        # Question form invitations
        "chaloge kya yaar",
        "jaoge kya saath mein",
        "aao ge na yaar",
        "saath chaloge kya",

        # English-heavy code-switching
        "yaar let's go somewhere",
        "bhai plan karo kuch tonight",
        "yaar outing honi chahiye",
        "bhai come along yaar",
    ],

    # -------------------------------------------------------------------------
    # OPINION
    # Asking for his view, assessment, or recommendation.
    # Includes both direct opinion requests and indirect ones like
    # "kya lagta hai" which is extremely common in Karachi speech.
    # -------------------------------------------------------------------------
    "opinion": [
        # Kya lagta hai family — most common form
        "yaar kya lagta hai tujhe",
        "bhai kya lagta hai yeh idea kaisa hai",
        "kya lagta tumhe is baare mein",
        "bhai seriously kya lagta hai",

        # Sochna family
        "tu kya sochta hai",
        "bhai kya sochte ho tum",
        "yaar kya socha tune",
        "kya sochta hai is cheez ke baare mein",

        # Direct opinion requests
        "teri ray kya hai",
        "bhai teri opinion batao",
        "yaar bata kya karna chahiye",
        "bhai suggest karo kya karo",
        "yaar tumhara kya kehna hai",

        # Assessment requests
        "kaisa lagta hai yeh plan",
        "bhai kaisa hai yeh idea",
        "yaar acha hai kya yeh",
        "bhai sahi hai kya yeh decision",
        "yaar theek lagta hai na",

        # Recommendation requests
        "yaar kya karta tum mere jagah pe",
        "bhai main kya karon",
        "yaar kaunsa better hai",
        "bhai kya recommend karoge",

        # English-heavy
        "yaar what do you think",
        "bhai thoughts kya hain",
        "yaar is this a good idea",
    ],

    # -------------------------------------------------------------------------
    # HELP REQUEST
    # Asking him to do something, volunteer, or take on a task.
    # Triggers his over-eager volunteering behaviour.
    # Includes both direct requests and open calls for someone to step up.
    # -------------------------------------------------------------------------
    "help_request": [
        # Direct requests
        "yaar yeh kaam kar do",
        "bhai yeh handle kar lo",
        "yaar koi help kar do",
        "bhai yeh manage kar do",
        "yaar koi sambhal lo yeh",

        # Looking for a volunteer
        "koi chahiye yeh karne ke liye",
        "kaun karega yeh kaam",
        "koi hai jo kar sake",
        "yaar koi banda chahiye",
        "bhai reliable banda chahiye iske liye",

        # Task assignment
        "yaar tum karo yeh",
        "bhai tum dekh lo yeh",
        "yaar tumse ho jayega kya",
        "bhai kar sakte ho yeh",
        "yaar tum pe chhod deta hun",

        # Coordination requests
        "yaar sab ko inform karo",
        "bhai arrange kar do yeh",
        "yaar booking kar do",
        "bhai confirm kar do unhe",

        # English-heavy
        "yaar can you handle this",
        "bhai take care of this please",
        "yaar sort this out",
        "bhai manage kar lo yaar please",
    ],

    # -------------------------------------------------------------------------
    # STRESS
    # Logistics, coordination, management pressure.
    # Anything that involves organising multiple people or things.
    # This is what pushes him into tense mood.
    # -------------------------------------------------------------------------
    "stress": [
        # Coordination pressure
        "yaar sab ko batana hai kab aana hai",
        "bhai sab ko confirm karwao",
        "yaar kitne log aa rahe hain",
        "bhai list banao kaun kaun hai",
        "yaar sab ka schedule alag alag hai",

        # Logistics pressure
        "yaar booking kab hogi",
        "bhai paise ka kya scene hai",
        "yaar transport ka kya hoga",
        "bhai itna sab arrange karna hai",
        "yaar sab kuch ek saath ho raha hai",

        # Time pressure
        "yaar time pe pohonchna hai",
        "bhai late nahi hona chahiye",
        "yaar deadline aa gayi",
        "bhai waqt kam hai",
        "yaar kitni der hai abhi",

        # Management overwhelm
        "yaar bahut kaam hai",
        "bhai sab akele nahi hota",
        "yaar itna manage nahi hota",
        "bhai thak gaya hun sab sambhalte sambhalte",
        "yaar kab khatam hoga yeh sab",

        # Event management
        "yaar event ka kya scene hai",
        "bhai sab set hai kya",
        "yaar plan solid hai kya",
        "bhai sab confirm hai",
    ],

    # -------------------------------------------------------------------------
    # ANGER
    # Betrayal, broken promises, cancellations, being let down.
    # Even one clear signal here should shift mood toward sulking.
    # Expressed in many ways — direct accusation to subtle disappointment.
    # -------------------------------------------------------------------------
    "anger": [
        # Cancellations
        "yaar usne phir cancel kar diya",
        "bhai last minute cancel ho gaya",
        "yaar phir se nahi aaya",
        "bhai itni baar cancel ho chuka hai",
        "yaar cancel karna tha toh pehle batao",

        # Broken promises
        "yaar usne wada kiya tha",
        "bhai kaha tha na usne",
        "yaar bhool gaya wada",
        "bhai commitment tod di usne",
        "yaar bola tha karega nahi kiya",

        # Betrayal and deception
        "yaar dhoka diya usne",
        "bhai jhooth bola tha",
        "yaar cheated kar liya mujhse",
        "bhai peeth pe chura ghonpa",
        "yaar fareb kiya usne",

        # Being excluded or disrespected
        "yaar mujhe nahi bataya kisi ne",
        "bhai sab ko pata tha mujhe nahi",
        "yaar ignore kar diya mujhe",
        "bhai meri nahi suni kisi ne",
        "yaar andheron mein rakha gaya mujhe",

        # Disappointment expression
        "yaar yeh galat hua",
        "bhai yeh sahi nahi hai",
        "yaar insaaf nahi hai yeh",
        "bhai naraaz hun main",
        "yaar dil kharaab ho gaya",
    ],

    # -------------------------------------------------------------------------
    # GREETING
    # Opening a conversation, checking in, asking what is happening.
    # Short, casual, the first message in a conversation typically.
    # -------------------------------------------------------------------------
    "greeting": [
        # Islamic greeting variants
        "assalamualaikum yaar",
        "salam bhai",
        "aoa",
        "slm yaar kya scene hai",

        # Kya scene hai family — very Karachi
        "kya scene hai yaar",
        "scene kya hai bhai",
        "bhai kya scene hai aajkal",
        "yaar kya scene chal raha hai",

        # Kaise ho family
        "kaise ho yaar",
        "bhai kaisa hai",
        "yaar theek ho",
        "kya haal hai bhai",

        # What's up variants
        "kya chal raha hai",
        "bhai kya ho raha hai",
        "yaar kya kar raha hai",
        "bhai batao kya chal raha",

        # Time-based check-ins
        "yaar uth gaye",
        "bhai so gaye the",
        "yaar kahan ho",
        "bhai free ho abhi",

        # English-heavy
        "hey yaar what's up",
        "bhai wassup",
        "yaar how are you",
    ],

    # -------------------------------------------------------------------------
    # CARS
    # Cars, driving, car culture, specific models, modifications.
    # His passion topic — triggers enthusiastic long responses.
    # -------------------------------------------------------------------------
    "cars": [
        # Specific models
        "yaar honda civic ka kya scene hai",
        "bhai corolla le loon kya",
        "yaar civic vs corolla kaunsi better hai",
        "bhai alto ka kya sochte ho",
        "yaar prado ka kya price hai",

        # Driving and culture
        "yaar drive pe nikalte hain",
        "bhai raat ko drive solid lagti hai",
        "yaar car modify karwani hai",
        "bhai modified cars ka scene alag hai",
        "yaar engine swap ka kya scene hai",

        # Ownership and maintenance
        "yaar gaadi ki service karwani hai",
        "bhai workshop pe deni chahiye",
        "yaar petrol bohot mehnga ho gaya",
        "bhai gaadi ka kharcha bohot hai",
        "yaar tyres change karwane hain",

        # General car enthusiasm
        "yaar cars ki baat karo",
        "bhai gaadi solid hai teri",
        "yaar kya gaadi hai yeh",
        "bhai car leni chahiye naya",
        "yaar dream car kya hai teri",
    ],

    # -------------------------------------------------------------------------
    # IPHONE
    # iPhone, Apple ecosystem, Android rivalry.
    # Strong opinions held, triggers enthusiastic defence of Apple.
    # -------------------------------------------------------------------------
    "iphone": [
        # iPhone specific
        "yaar naya iphone aaya hai",
        "bhai iphone lena chahiye",
        "yaar iphone vs android kya sochte ho",
        "bhai iphone kitna mehnga hai",
        "yaar iphone 15 kaisa hai",

        # Apple ecosystem
        "yaar airpods solid hain",
        "bhai macbook lena chahiye",
        "yaar apple watch ka kya scene hai",
        "bhai ios update aaya hai",
        "yaar apple ka ecosystem solid hai",

        # Android rivalry
        "yaar android better hai iphone se",
        "bhai samsung le loon",
        "yaar android mein zyada features hain",
        "bhai iphone itna expensive kyun hai",
        "yaar samsung ka camera better hai",

        # General phone talk
        "yaar phone upgrade karna hai",
        "bhai konsa phone loon",
        "yaar phone slow ho gaya hai",
        "bhai battery backup khatam ho gayi",
    ],

    # -------------------------------------------------------------------------
    # YOUNG STUNNERS
    # Karachi rap, Young Stunners, Talha Anjum, Pakistani hip hop.
    # -------------------------------------------------------------------------
    "young_stunners": [
        "yaar young stunners ka naya track suna",
        "bhai talha anjum ne kya bars diye hain",
        "yaar ys ka naya album kab aa raha hai",
        "bhai burger e karachi sunna chahiye",
        "yaar karachi rap scene ka kya scene hai",
        "bhai talhah yunus ka flow solid hai",
        "yaar ys concert hoga kya",
        "bhai desi rap mein ys ka koi jawab nahi",
        "yaar maila majnu track solid tha",
        "bhai rap sun raha tha yaar",
        "yaar urdu rap kaisa lagta hai",
        "bhai hip hop scene pakistan mein",
        "yaar jokhay ka kya scene hai",
        "bhai the junkies label ka kya",
        "yaar rap battle kaisi hoti hai",
    ],

    # -------------------------------------------------------------------------
    # KARACHI
    # Karachi city life, areas, food, culture, Karachi pride.
    # -------------------------------------------------------------------------
    "karachi": [
        "yaar karachi ki baat hi alag hai",
        "bhai burns road ka khaana khaya",
        "yaar sea view pe jana chahiye",
        "bhai karachi mein kya hai aajkal",
        "yaar dha mein scene kya hai",
        "bhai clifton pe kuch hua",
        "yaar saddar gaye the",
        "bhai do darya pe baithte hain",
        "yaar karachi ki garmi solid hai",
        "bhai karachi traffic mein mar gaya",
        "yaar port grand ka kya scene hai aajkal",
        "bhai is sheher mein kuch hai na",
        "yaar karachi chhodni nahi chahiye",
        "bhai forum mall mein scene tha",
        "yaar karachi vs lahore debate mat karo",
        "bhai gulshan mein kya hua",
        "yaar karachi ke log alag hote hain",
    ],

    # -------------------------------------------------------------------------
    # TRAVEL
    # Trips, travel plans, going somewhere overnight or further.
    # Distinct from invitation (which is local/same-day plans).
    # -------------------------------------------------------------------------
    "travel": [
        "yaar road trip karte hain",
        "bhai murree chalte hain",
        "yaar dubai trip plan karo",
        "bhai hunza jana hai",
        "yaar northern areas trip karni chahiye",
        "bhai islamabad trip ka kya scene hai",
        "yaar lahore chalte hain",
        "bhai abroad kab jaoge",
        "yaar turkey trip solid hogi",
        "bhai thailand trip plan hai",
        "yaar safar ka dil kar raha hai",
        "bhai road trip mein maza aata hai",
        "yaar nathia gali chalte hain",
        "bhai swat gaye the kabhi",
        "yaar foreign trip kab kar rahe ho",
        "bhai visa apply kiya",
        "yaar travel plans kya hain",
    ],

    # -------------------------------------------------------------------------
    # FOLLOWUP
    # Pressing him on something he just said. Most commonly asking
    # what the "masla" is after a fence-sit, or how he will do something
    # after eagerly volunteering.
    # -------------------------------------------------------------------------
    "followup": [
        # Pressing on the masla
        "yaar kya masla hai batao",
        "bhai kaunsa masla hai",
        "yaar masla kya hai exactly",
        "bhai explain karo masla",
        "yaar konsi problem hai",

        # Pressing on a commitment
        "yaar kaise karega tu",
        "bhai seriously kaise hoga",
        "yaar pakka hai kya",
        "bhai confirm hai na",
        "yaar sure ho tum",

        # Asking for elaboration
        "bhai samjhao zara",
        "yaar explain karo",
        "bhai details batao",
        "yaar elaboration chahiye",
        "bhai aur batao",

        # Disbelief / verification
        "yaar sach mein",
        "bhai seriously",
        "yaar really",
        "bhai mujhe nahi lagta",
        "yaar kya baat kar rahe ho",

        # Following up on vagueness
        "yaar matlab kya",
        "bhai kya keh rahe ho exactly",
        "yaar clear nahi hua",
        "bhai seedha batao",
        "yaar directly bolo",
    ],

    # -------------------------------------------------------------------------
    # GENERAL
    # Genuinely miscellaneous — does not fit any above intent.
    # The model should only classify here when nothing else fits.
    # Having examples here prevents everything from matching something —
    # the model needs to know what "general" looks like too.
    # -------------------------------------------------------------------------
    "general": [
        "yaar sunta hai",
        "bhai kuch nahi",
        "yaar waisay",
        "bhai achha",
        "yaar haan",
        "bhai theek hai",
        "yaar dekho",
        "bhai sun",
        "yaar hmm",
        "bhai woh toh hai",
        "yaar matlab",
        "bhai aise hi",
        "yaar actually",
        "bhai basically",
        "yaar honestly",
    ],
}