dea 1: Self-Evolving Image Editing in Specialized Image Editing Models (Qwen-Image-Edit)
The key architectural observation here is specific to Qwen-Image-Edit's dual-path design:
Qwen-Image-Edit already has Qwen2.5-VL baked in as an internal understanding encoder, but jointly trained as part of the same model
The understanding branch can act as an internal Proposer (generating editing instructions from raw unlabeled images) and simultaneously as an internal solver (scoring whether the edited output matches the instruction and whether unchanged regions are preserved)
The generative MMDiT backbone acts as the Editor
The entire loop is closed and no external model call at any stage

But htis only works for models which have image understanding capabilities:
Pure diffusion models like FLUX, SD3, and standard DiT have no internal understanding pathway so any self-improvement loop there necessarily pulls in something external
Unified models like BAGEL and OmniGen2 also qualify architecturally but then its becomes unified models problem.

Current gap in literature:
Edit-R1 uses MLLM logits as reward but from a separately loaded external model
NP-Edit uses VLM gradient feedback from an external VLM
UniEdit-I introduces an understanding–editing–verifying loop for unified VLMs, but it is training-free rather than self-evolving post-training.
I have not found prior work that explicitly uses Qwen-Image-Edit’s own internal dual-path design to build a fully closed, weight-updating self-evolving editing loop without any external reward model.
 
Core technical contributions would be:
a Proposer-Editor-Solver loop entirely within Qwen-Image-Edit
A difficulty shaping mechanism for the Proposer (similar to our eccv paper) to prevent instruction collapse toward trivially easy edits
A decomposed internal reward with two components:
-Global reward: does the model's description of the edited image match the instruction
-Local reward: do regions that should be unchanged remain consistent with the original
Showing that the loop works iteratively with each round produces a stronger Editor, which enables the Proposer to generate harder instructions, creating a genuine curriculum without any human design
 
Key challenges:
the model could learn edits that score well internally but are visually poor
the Proposer may collapse to trivially simple instructions without difficulty shaping
fine-grained edits may exceed the understanding module's verification resolution
 