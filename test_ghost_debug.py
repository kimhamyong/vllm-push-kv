#!/usr/bin/env python3
"""Send sequential requests through push proxy and identify ghost responses."""
import requests
import time
import json
import sys

PROXY_URL = "http://127.0.0.1:8000/v1/completions"
OUTPUT_LEN = 200

_base_prompts = [
    "The quick brown fox jumps over the lazy dog and then runs across the wide open field where many animals gather to play and rest under the warm afternoon sun while birds sing their beautiful songs in the tall green trees nearby. The river flows gently through the valley carrying fallen leaves and small twigs downstream toward the distant ocean where waves crash upon the rocky shore. Fishermen cast their nets into the deep blue water hoping for a bountiful catch while seagulls circle overhead calling to one another. The clouds drift slowly across the sky painting shadows on the landscape below as the day progresses from morning to afternoon. Children play in the meadow chasing butterflies and picking wildflowers to bring home to their families. The old stone bridge arches gracefully over the stream connecting the two villages that have traded goods for centuries. Farmers tend their crops in the fertile fields watching the weather for signs of rain that will nourish the growing plants. The forest at the edge of town is home to deer foxes rabbits and countless species of birds that fill the air with music at dawn.",
    "In a galaxy far far away there existed a civilization of advanced beings who had mastered the art of interstellar travel and communication across vast distances using quantum entanglement technology that allowed them to share knowledge instantly across light years. Their ships were powered by antimatter engines capable of bending spacetime itself creating stable wormholes between star systems. The civilization had colonized thousands of worlds each with its own unique ecosystem and culture but all connected through a vast neural network that spanned the galaxy. Scientists on the homeworld continued to push the boundaries of physics discovering new dimensions of reality that challenged everything they thought they knew about the universe. The council of elders governed wisely balancing the needs of trillions of citizens spread across countless planets moons and space stations. Artists created works that could only be experienced in zero gravity while musicians composed symphonies using the electromagnetic frequencies of pulsars and magnetars. Engineers built megastructures around dying stars harvesting their final energy output to power the civilization for millennia to come.",
    "Once upon a time there was a young wizard who discovered an ancient book of spells hidden deep within the forbidden library of the grand castle where generations of powerful sorcerers had studied and practiced their magical arts for over a thousand years. The book was bound in dragon leather and its pages were made from enchanted parchment that could only be read by moonlight. Each spell within was more powerful and dangerous than the last requiring immense concentration and magical energy to cast properly. The young wizard spent months studying the first chapter alone learning the fundamental principles of elemental manipulation and dimensional folding. The castle itself was alive with magic its walls shifting and corridors rearranging themselves according to ancient enchantments placed by the founders. Ghosts of former students wandered the halls offering cryptic advice to those brave enough to listen. The library contained millions of books scrolls and artifacts collected from every corner of the known world and several corners of worlds unknown. Deep beneath the castle lay a network of caverns where underground rivers of pure magical energy flowed providing power to the wards and enchantments that protected the school.",
    "The meaning of life is a philosophical question that has been debated by great thinkers throughout human history from ancient Greek philosophers like Socrates and Plato to modern existentialists who explored the nature of consciousness and purpose in an apparently indifferent universe. Eastern traditions offer perspectives centered on mindfulness compassion and the interconnectedness of all living things suggesting that meaning arises from our relationships with others and the natural world. Scientific discoveries have revealed the astonishing complexity of biological systems from the molecular machinery of cells to the emergent properties of consciousness in the human brain raising profound questions about free will determinism and the nature of subjective experience. Some argue that meaning is inherent in the structure of reality itself encoded in mathematical laws and physical constants that seem remarkably fine tuned for the emergence of complex life. Others maintain that meaning is a human construction something we create through our choices actions and commitments rather than something we discover. The existentialist tradition emphasizes radical freedom and responsibility arguing that we are condemned to be free and must create our own values in a world without predetermined purpose.",
    "Artificial intelligence will transform every aspect of modern society including healthcare education transportation manufacturing and entertainment as machine learning algorithms become increasingly sophisticated and capable of solving complex problems that were previously thought to require human intelligence and creativity. In medicine AI systems can analyze medical images with superhuman accuracy detecting cancers tumors and other abnormalities that human radiologists might miss leading to earlier diagnosis and better patient outcomes. Autonomous vehicles powered by deep learning neural networks will revolutionize transportation reducing accidents caused by human error and providing mobility to elderly and disabled populations who currently cannot drive. In education personalized AI tutors will adapt to each students learning style pace and interests providing customized instruction that maximizes engagement and knowledge retention. Manufacturing will be transformed by intelligent robots that can learn new tasks through observation and practice rather than requiring explicit programming for every movement and decision. Creative industries will see AI tools that assist human artists musicians and writers generating novel ideas and helping to explore vast creative spaces that would be impossible to navigate manually.",
    "San Francisco is known for its iconic Golden Gate Bridge steep rolling hills historic cable cars and vibrant cultural diversity that attracts millions of visitors from around the world who come to experience its unique blend of technology and tradition art and innovation natural beauty and urban sophistication. The city was founded during the California Gold Rush of 1849 when thousands of prospectors flooded into the area seeking their fortune in the rivers and mountains of the Sierra Nevada. Today it stands as the heart of Silicon Valley the global center of technological innovation where companies like Apple Google Meta and countless startups continue to push the boundaries of what technology can achieve. The citys neighborhoods each have their own distinct character from the bohemian atmosphere of Haight Ashbury to the vibrant Chinatown the largest outside of Asia to the trendy restaurants and boutiques of the Mission District. Alcatraz Island sitting in the cold waters of the bay once housed Americas most notorious criminals and now serves as one of the citys most popular tourist attractions. The fog that rolls in from the Pacific Ocean each evening gives the city an ethereal quality transforming familiar landmarks into mysterious silhouettes.",
    "The best programming language is a topic of endless debate among software developers who argue passionately about the merits of Python Java Rust Go and many other languages each designed to solve different types of computational problems efficiently and elegantly. Python has emerged as the dominant language for data science machine learning and artificial intelligence thanks to its clean syntax extensive library ecosystem and gentle learning curve that makes it accessible to beginners while remaining powerful enough for experts. Rust has gained tremendous popularity for systems programming offering memory safety without garbage collection through its innovative ownership and borrowing system that catches entire categories of bugs at compile time rather than runtime. Go designed at Google provides excellent concurrency primitives and compiles to fast native code making it ideal for building scalable network services and cloud infrastructure. JavaScript continues to dominate web development running in every browser and increasingly on servers through Node.js creating a unified language ecosystem for full stack development. Each language represents different design tradeoffs and philosophies reflecting the diverse needs of the software industry from embedded systems to web applications from scientific computing to game development.",
    "Machine learning models can analyze vast amounts of data to discover hidden patterns and make accurate predictions that would be impossible for humans to detect manually enabling breakthroughs in medical diagnosis financial forecasting scientific research drug discovery climate modeling and many other fields that impact human welfare. Deep neural networks with billions of parameters trained on massive datasets have achieved remarkable performance on tasks ranging from image classification and object detection to natural language understanding and generation. Transfer learning allows models pretrained on large general datasets to be fine tuned for specific tasks with relatively small amounts of labeled data dramatically reducing the cost and time required to develop specialized AI applications. Reinforcement learning has produced agents capable of superhuman performance in complex games like Go chess and video games learning optimal strategies through millions of simulated episodes of trial and error. Generative models including variational autoencoders and generative adversarial networks can create realistic images videos music and text opening new possibilities for creative expression and content production. The field continues to advance rapidly with new architectures training techniques and theoretical insights emerging at an accelerating pace.",
    "The future of technology holds incredible promise with advances in quantum computing biotechnology renewable energy artificial intelligence and space exploration that will fundamentally change how humans live work communicate and understand the universe around them. Quantum computers leveraging the strange properties of superposition and entanglement will solve problems that are intractable for classical computers including drug design materials science optimization and cryptography. CRISPR gene editing technology has given scientists unprecedented ability to modify DNA with precision opening possibilities for curing genetic diseases eliminating invasive species and engineering crops that can withstand climate change. Fusion energy the process that powers the sun is finally approaching commercial viability after decades of research promising virtually unlimited clean energy that could end humanitys dependence on fossil fuels and dramatically reduce greenhouse gas emissions. Brain computer interfaces are advancing rapidly with companies developing implantable devices that could restore movement to paralyzed patients treat neurological disorders and eventually enhance human cognitive capabilities. Space agencies and private companies are planning permanent settlements on the Moon and Mars beginning a new chapter in human history as a multiplanetary species.",
    "Deep learning networks are composed of multiple layers of interconnected neurons that process information hierarchically extracting increasingly abstract features from raw data to perform tasks such as image recognition natural language understanding and autonomous navigation with remarkable accuracy and generalization capability. The transformer architecture introduced in 2017 revolutionized natural language processing by using self attention mechanisms that allow the model to weigh the importance of different parts of the input when generating each element of the output. Large language models built on the transformer architecture have demonstrated emergent capabilities including reasoning planning code generation and even creative writing that were not explicitly trained for but arise from the scale of the model and training data. Convolutional neural networks remain the backbone of computer vision processing images through layers of learned filters that detect edges textures shapes and objects at increasing levels of abstraction. Recurrent architectures including LSTMs and GRUs are designed to process sequential data maintaining hidden states that capture temporal dependencies in time series speech and other dynamic signals. The field of neural architecture search uses AI itself to design optimal network architectures automatically discovering configurations that outperform human designed models on benchmark tasks."
]

prompts = [_base_prompts[i % len(_base_prompts)] for i in range(20)]

results = []
ghosts = []

for i, prompt in enumerate(prompts):
    payload = {
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": prompt,
        "max_tokens": OUTPUT_LEN,
        "temperature": 0.0,
    }

    t0 = time.time()
    try:
        resp = requests.post(PROXY_URL, json=payload, timeout=120)
        elapsed = time.time() - t0
        data = resp.json()

        if "choices" in data and len(data["choices"]) > 0:
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            text = data["choices"][0].get("text", "")

            is_ghost = completion_tokens == 0
            tpot = (elapsed / completion_tokens * 1000) if completion_tokens > 0 else 0.0

            result = {
                "req_idx": i,
                "prompt_idx": i % 10,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": round(elapsed * 1000, 1),
                "tpot_ms": round(tpot, 2),
                "is_ghost": is_ghost,
                "text_preview": text[:80] if text else "(empty)",
            }
            results.append(result)

            status = "GHOST!" if is_ghost else "OK"
            print(f"[{status}] req {i:2d} (prompt {i%10}): "
                  f"completion={completion_tokens:3d} tokens, "
                  f"elapsed={elapsed*1000:.0f}ms, "
                  f"tpot={tpot:.1f}ms")

            if is_ghost:
                ghosts.append(result)
        else:
            print(f"[ERROR] req {i}: unexpected response: {json.dumps(data)[:200]}")
            results.append({"req_idx": i, "error": True})

    except Exception as e:
        print(f"[ERROR] req {i}: {e}")
        results.append({"req_idx": i, "error": str(e)})

print(f"\n{'='*60}")
print(f"Total: {len(results)} requests, {len(ghosts)} ghosts")
if ghosts:
    print(f"Ghost requests: {[g['req_idx'] for g in ghosts]}")
    print(f"Ghost prompt indices: {[g['prompt_idx'] for g in ghosts]}")
else:
    print("No ghosts detected!")

# Save results
with open("/tmp/ghost_debug_results.json", "w") as f:
    json.dump({"results": results, "ghosts": ghosts}, f, indent=2)
print(f"\nResults saved to /tmp/ghost_debug_results.json")
