// Run the in-browser engine on the three example contexts and print JSON, so
// sim/capture.py can check engine == fieldrun. Node-only helper.
const fs = require('fs');
const E = require('./engine.js');
const w = JSON.parse(fs.readFileSync('sim/data/weights.json'));
const lex = JSON.parse(fs.readFileSync('sim/data/lexicon.json'));
const store = JSON.parse(fs.readFileSync('sim/data/store.json'));
const m = E.loadModel(w);
const out = [];
for (const ex of lex.examples) {
  const o = E.forward(m, ex.prefix);
  const r = E.route(store, ex.prefix, o.pred);
  out.push({ key: ex.key, pred: o.pred, logit: o.logits[o.pred], margin: o.margin,
             normMargin: o.normMargin, PR: o.PR, route: r.route, idiom: r.idiom,
             covered: r.covered });
}
process.stdout.write(JSON.stringify(out));
