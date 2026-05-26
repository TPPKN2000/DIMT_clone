const MathJax = require('mathjax');
const readline = require('readline');

MathJax.init({
  loader: { load: ['input/tex', 'output/svg', 'adaptors/liteDOM'] }
}).then((mj) => {
  const adaptor = mj.startup.adaptor;
  
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
  });

  // Output ready signal to stdin of caller
  console.log(JSON.stringify({ status: "ready" }));

  rl.on('line', async (line) => {
    if (!line.trim()) return;
    try {
      const req = JSON.parse(line);
      const tex = req.tex;
      const display = req.display !== false;
      
      // Use promise-based API to handle async actions (e.g. \mathfrak)
      const node = await mj.tex2svgPromise(tex, { display: display });
      const svgNodes = adaptor.tags(node, 'svg');
      if (!svgNodes || svgNodes.length === 0) {
        console.log(JSON.stringify({ error: "No SVG output produced for: " + tex }));
        return;
      }
      const svg = adaptor.serializeXML(svgNodes[0]);
      
      console.log(JSON.stringify({ id: req.id, svg: svg }));
    } catch (err) {
      console.log(JSON.stringify({ error: err.message }));
    }
  });
}).catch((err) => {
  console.error(JSON.stringify({ status: "error", error: err.message }));
  process.exit(1);
});
