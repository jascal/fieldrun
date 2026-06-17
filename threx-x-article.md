# The Threx

> Draft to post on X (as an Article, or as the thread at the bottom).
> **Link to confirm before posting:** https://jascal.github.io/fieldrun/intuition.html
> (default GitHub Pages URL for this repo — swap in your real URL if different).

---

## Article

Somewhere in the dark of an alien sea live the **Threx**: blind foragers who have never once seen another of their kind. Everything they coordinate — finding food, warning of danger, splitting a catch — has to travel as sound. So they speak in **pulses**: short, terse strings called *calls*. A wasted pulse is energy a Threx doesn't have, so the whole language is ruthlessly compressed — just **25 pulses**, and seven sentence patterns that cover everything they will ever need to say.

A few, to get your ear in:

- `⟨ ka na dø ⟩` — *we · warn! · deep.* A danger cry: something is below.
- `⟨ fa · tu wø bo ⟩` — *far · you · seek · berry.* Go forage berries, out there.
- `⟨ ∿ ↓ → … ⟩` — *triangulate · south-current · east-current · ?* Where those two currents cross, you'll find your prey — but which prey?

The Threx aren't real. I invented them — under one strict constraint. I wanted a language where simply **predicting the next pulse** forces three genuinely different kinds of thinking:

- **Recall.** The danger cry is fixed ritual: after *warn*, the only pulse that ever follows is *deep*. You don't reason about it — you remember it.
- **Selection.** *Seek* opens a known menu — fish, berry, shell — but which one depends on the *place* named several pulses back. The set is memorized; the choice is contextual.
- **Computation.** Prey lies where two currents cross. To name it you must take two directions, **add** them, and read off what lives there. The answer is written in the call nowhere — it has to be worked out. And some current-pairs are never shown in training, so memorizing can't save you.

That's the whole reason the Threx exist: a tiny, fully-understood world where **memory and computation are pulled apart on purpose**, so you can tell which one is doing the work.

Then I trained a real language model on their corpus — a from-scratch RoPE/Llama-style transformer, small enough to watch end to end — and built a page where you can run it yourself, pulse by pulse.

You can step through its attention and its neurons, watch it write a whole call from a single prompt, and hover any block for plain English. The part I like best: a strict **phrasebook** baseline (pure memorization — n-grams, recency, a copy rule) runs alongside the model and tries to keep up. It nails the rituals. It often picks the right item off a menu. And it *fails* — every time — exactly when the Threx force the model to compute.

Meet the Threx, and watch a tiny mind learn to answer them:

→ **https://jascal.github.io/fieldrun/intuition.html**

---

## Thread version (if you'd rather post a thread)

**1/** Meet the Threx: blind foragers in an alien sea who have never seen each other. They coordinate entirely in *pulses* — short calls of sound. Their whole language is just 25 pulses. I invented them for one reason 🧵

**2/** I wanted a language where predicting the next pulse forces three different kinds of thinking:
• RECALL — after *warn*, the next pulse is always *deep*
• SELECTION — *seek* → {fish, berry, shell}, context picks which
• COMPUTATION — prey = where two currents cross. You have to *add* them.

**3/** The point of the Threx: a tiny, fully-understood world where memory and computation are pulled apart on purpose — so you can see which one a model is actually using.

**4/** Then I trained a real (from-scratch, RoPE/Llama-style) language model on Threx and made it watchable, pulse by pulse: attention, neurons, the whole forward pass, and it writing a full call end to end.

**5/** Best part: a pure-memorization "phrasebook" runs alongside and tries to keep up. It aces the rituals — and fails *exactly* when the Threx force the model to compute. Meet them here:
→ https://jascal.github.io/fieldrun/intuition.html
