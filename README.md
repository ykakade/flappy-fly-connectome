# Flappy Fly

Play Flappy Bird, or watch a controller trained with connections from a fruit fly brain.

[Play online](https://flappy-fly-connectome.pox.chatgpt.site)

## Play

Press **Space** or **↑**, click, or tap to flap. Get through the pipes without hitting them or the ground. Press **R** to restart.

The game has the classic pixel bird, green pipes, scrolling ground, sounds, and score display. Your best score stays in your browser.

## Try the computer

- **Watch:** pick a controller and watch it play.
- **You vs. computer:** play the same pipe layout together. You are yellow. The computer is blue.
- **Training:** see how the controller played before and after training.
- **Change the wiring:** cut connections or change their strengths, then compare the result.

The computer modes use the easier rules the saved controllers were trained on. Play uses separate classic rules. Their best scores are saved separately.

## What is the fly part?

The controller uses 96 neurons and 602 connections from the [MaleCNS dataset](https://male-cns.janelia.org/download/). It reads the bird’s position, speed, and the next pipe opening, then decides whether to flap. It does not look at the screen.

After training, it passed **16.88 pipes on average** across 32 new layouts. Before training, it averaged zero. Shuffled wiring scored 18.44, and a standard neural network scored 18.16. These results do not show an advantage for fly wiring.

This is a small model built from brain wiring data, not a simulation of a living fly.

## Run locally

You need Node.js 20 or newer and Python 3. There are no npm packages to install.

```sh
git clone --branch master https://github.com/ykakade/flappy-fly-connectome.git
cd flappy-fly-connectome
npm run build
npm run dev
```

Open [localhost:4173](http://localhost:4173).

To check the browser code:

```sh
npm run lint
npm test
```

## More details

- [Experiment notes and training instructions](docs/experiment.md)
- [Data sources and processing](data/README.md)
- [Full results](metrics/evaluation.json)

The website lives in `dist/`. Training code lives in `src/flappy_fly/`. The saved models live in `checkpoints/`.

## Credits

Flappy Bird was created by Dong Nguyen. Sprites and sounds come from [Samuel Custodio’s asset collection](https://github.com/samuelcust/flappy-bird-assets), with its [license](dist/assets/LICENSE) included. This is an unofficial recreation.

Brain data: FlyEM at HHMI Janelia, University of Cambridge, MRC Laboratory of Molecular Biology, and Google Research. The selected graph uses [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See [CITATION.cff](CITATION.cff) for the paper.

Project code: [MIT](LICENSE).
