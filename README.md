# CyberNoodles: Early Access

Use this to help implement CyberNoodles into your projects, with details down below to help you out as much as I possibly can :D

## What do I use this for?

Test your projects and fun little prototypes with my model that is going to play Beat Saber (at some point)

## How you would use it

pip install -r requirements.txt # Install the stuff from the requirements.txt
python generate_replay.py 464f2 # Generate a replay with the BSR Code 464f2.
python generate_replay.py 464f2 --difficulty ExpertPlus # Generate that replay, but specify ExpertPlus!
python generate_replay.py 464f2 --max-windows 40 # Generate a certain amount of replay (40 Windows = ~0.25 seconds?)

It will output your good old standard BSOR :)

## I see a bunch of onnx models, what the hell do they do?!

`map_encoder.onnx`  stats (B,6), notes (B,50,31), walls (B,50,6), past frames (B,50,22), conditioning tokens (B,152,512) + note embeddings (B,50,512) 

`masked_prior.onnx` token ids (B,50) int64, visible mask (B,50) bool, frame times (B,50), conditioning (B,152,512), per-slot code logits (B,50,512) 

`residual.onnx` code stack (B,8,50) int64, stacked head logits (B,7,50,512) 

`tok_decode.onnx` codes (B,50,8) int64, motion frames (B,50,21)

`hit_scorer.onnx` note embeddings (B,50,512), motion (B,50,21), per-note hit logits (B,50,7)

## The control knobs (I worked hard on these so they matter)

- `--temp 0.8` -- Your standard temp knob, let the model go crazy or make it chill? You choose XD
- `--cfg 3.0` -- Control how much the model either relies on Map Context, or the generic motion prior, this doesn't matter much, use default.
- `--iters 12` -- How refined do you want the replay? (More iteration = Slower generation, use sparingly)
- `--hit-cands 2` -- How many candidates do you want to use? Explore better accuracy with more, at the cost of speed.
- `--r-temp 0` -- How random do you want your residual model?

## Porting to other runtimes (JS/C#)

Look at the Python Implementation, I tried my best to make it somewhat easy to understand and modular, try your best, you are the real coder here...

## What is in here?

- `generate_replay.py` -- Generate a replay!
- `onnx/` -- Where the models are stored
- `lib/` -- Where the libs are stored
- `maps/` -- Map cache, so you don't download the same map over and over

## Known limitations of this build (Boring stuff)

- THESE WEIGHTS AIN'T TRAINED GOOD, don't expect some CyberRamen type play, CyberNoodles is just a baby ;(
- I want the model to support a buncha types (v3, NE, ME...) but idk if it's good enough, will try to improve :D
- The weights are kinda heavy compared to CyberRamen, will try to optimize but don't expect anything good.
