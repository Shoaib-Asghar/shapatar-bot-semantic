# =============================================================================
# test_dataset.py
# =============================================================================
# 
# Holdout test dataset for evaluating Shapatar Bot's semantic embeddings.
# These phrases do NOT exist in intent_examples.py. 
# Testing on these ensures the bot is actually "understanding" the meaning
# and generalizing to new phrases, rather than just memorizing its training data.
# =============================================================================

TEST_DATA = {
    "invitation": [
        "kidhar nikalna hai phir aaj",
        "bhai koi scene on karo yaar",
        "bahar milte hain raat ko",
        "plan done karo jaldi se",
        "aaj ka kya program hai"
    ],
    "opinion": [
        "batao kaisa laga tumhein",
        "tera kya review hai is pe",
        "achha idea hai na yeh?",
        "mashwara chahiye tha ek",
        "kya sahi rahega mere liye"
    ],
    "help_request": [
        "bhai aik kaam tha tujhse",
        "yaar thori madad kardo isme",
        "kya tu yeh handle kar lega",
        "kisi ko bolo yeh sambhalne ke liye",
        "yeh kon karega yaar"
    ],
    "stress": [
        "sab kuch sar par aa gaya hai",
        "kaun kaun aayega list final karo",
        "bhai budget manage nahi ho raha",
        "bohot pressure hai yaar aajkal",
        "timing fix nahi ho rahi kisi ki"
    ],
    "anger": [
        "tune dhoka diya yaar",
        "wada karke nahi aya tu",
        "cancel kyun kiya last moment",
        "mujhe andheron mein kyun rakha gaya",
        "bhai yeh umeed nahi thi tujhse"
    ],
    "greeting": [
        "kia scene hai jani",
        "kya haal hain tere",
        "assalam o alaikum bhai",
        "aur sunao sab theek?",
        "uth gaya tu?"
    ],
    "cars": [
        "bhai civic turbo ka kya rate hai aajkal",
        "mehran modify karni hai thori",
        "tyre pressure kitna rakhte ho",
        "gaadi ki average bohot kam hai",
        "engine sound kar raha hai yaar"
    ],
    "iphone": [
        "yaar 15 pro max le lun?",
        "android ki battery better hoti hai",
        "ios 18 update kar lia kya",
        "airpods ki sound solid hai",
        "apple ka ecosystem hi alag hai bhai"
    ],
    "young_stunners": [
        "anjum ka naya gana suna tune?",
        "karachi rap king ys hai",
        "jokhay ki beat bohot tight thi bhai",
        "burger e karachi repeat pe chal raha hai",
        "talhah yunus ka flow check kar"
    ],
    "karachi": [
        "bhai clifton ki taraf traffic full block hai",
        "tariq road pe rush bohot hai aaj",
        "saddar se maal uthana hai",
        "karachi ka mausam tight hai aaj",
        "do darya pe khana acha tha"
    ],
    "travel": [
        "bhai nathiagali ka plan ban raha hai",
        "dubai ka visa lag gaya mera",
        "road trip ka mood ho raha hai north ki taraf",
        "aglay mahine london ja raha hun",
        "ticket kitne ka mil raha hai"
    ],
    "followup": [
        "bhai clear bata na",
        "masla kidhar aa raha hai exactly",
        "wajah toh bata de yaar",
        "seriously karega ya nahi",
        "matlab phir kya socha hai"
    ],
    "general": [
        "acha theek hai dekh lenge",
        "haan pata nahi",
        "wese hi bol raha tha",
        "haan sahi keh raha hai",
        "chalo baad mein dekhte hain"
    ]
}
