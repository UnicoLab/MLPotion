# MLPotion: Brew Your ML Magic! 🧪✨

<p align="center">
  <img src="logo.png" width="350" alt="MLPotion Logo"/>
  <p align="center"><strong>Provided and maintained by <a href="https://unicolab.ai">🦄 UnicoLab</a></strong></p>
</p>

Welcome, fellow alchemist! 🧙‍♂️ Ready to brew some machine learning magic without getting locked in a cauldron?

**MLPotion** is your chest of modular, mix-and-match ML building blocks that work across **Keras, TensorFlow, and PyTorch**. Think of it as LEGO® for ML pipelines, but with fewer foot injuries and more flexibility!

## Why MLPotion? 🤔

Ever felt trapped by a framework that forces you to do things "their way"? We've been there. That's why we created MLPotion:

- **🎯 Framework Agnostic**: Write once, run anywhere (well, on Keras, TensorFlow, or PyTorch)
- **🧱 Modular by Design**: Pick the pieces you need, leave the rest in the box
- **🔬 Type-Safe**: Python 3.10+ typing that actually helps you (mypy approved!)
- **🚀 Production Ready**: Built for the real world, not just notebooks
- **🎨 Orchestration Flexible**: Works standalone OR with ZenML, Prefect, Airflow - your choice!
- **📦 Install What You Need**: Core package works without any ML frameworks (you only install what you need)!
- **🤝 Community-Driven**: Missing something? Contribute it back - we love community additions!

## What's in the Potion? 🧪

<div class="grid-container">
  <div class="grid-item">
    <h3>⚗️ Core Ingredients</h3>
    <ul class="feature-list">
      <li>Type-safe protocols for all components</li>
      <li>Framework-agnostic result types</li>
      <li>Consistent error handling</li>
      <li>Zero-dependency core package</li>
    </ul>
  </div>

  <div class="grid-item">
    <h3>🔧 Framework Support</h3>
    <ul class="feature-list">
      <li><strong>Keras 3.0+</strong> - The friendly one</li>
      <li><strong>TensorFlow 2.15+</strong> - The production workhorse</li>
      <li><strong>PyTorch 2.0+</strong> - The researcher's favorite</li>
    </ul>
  </div>

  <div class="grid-item">
    <h3>📊 Data Processing</h3>
    <ul class="feature-list">
      <li>CSV loaders for all frameworks</li>
      <li>Dataset optimization utilities</li>
      <li>Data transformers</li>
      <li>Preprocessing pipelines</li>
    </ul>
  </div>
</div>

<div class="grid-container">
  <div class="grid-item">
    <h3>🎓 Training & Evaluation</h3>
    <ul class="feature-list">
      <li>Unified training interface</li>
      <li>Comprehensive evaluation tools</li>
      <li>Rich result objects</li>
      <li>Training history tracking</li>
    </ul>
  </div>

  <div class="grid-item">
    <h3>💾 Model Management</h3>
    <ul class="feature-list">
      <li>Save/load model checkpoints</li>
      <li>Export to production formats</li>
      <li>Model inspection utilities</li>
      <li>Multiple export formats</li>
    </ul>
  </div>

  <div class="grid-item">
    <h3>🔄 Orchestration Integration</h3>
    <ul class="feature-list">
      <li>ZenML integration built-in</li>
      <li>Extensible to Prefect, Airflow, etc.</li>
      <li>Works standalone (no orchestration needed!)</li>
      <li>Community contributions welcome</li>
    </ul>
  </div>
</div>

## The MLPotion Philosophy 🎭

> "A good potion doesn't force you to drink it a certain way. It just... works."
>
> — Ancient ML Alchemist Proverb (we just made that up)

We believe in:

1. **Flexibility > Convention**: Your project, your rules
2. **Simplicity > Complexity**: If it's hard to use, we failed
3. **Type Safety > Runtime Surprises**: Catch errors before they bite
4. **Modularity > Monoliths**: Use what you need, ignore the rest
5. **Consistency > Chaos**: Same patterns across all frameworks
6. **Community > Corporate**: Built by the community, for the community

## Extensibility & Community Contributions 🌟

MLPotion is designed to be **extensible**. While we provide ZenML integration out-of-the-box, you can easily integrate with:

- **Prefect**: Wrap components as Prefect tasks
- **Airflow**: Use as operators in DAGs
- **Kubeflow**: Deploy in Kubeflow pipelines
- **Your Custom Orchestrator**: The building blocks work anywhere!

**Missing a feature?** We actively encourage community contributions! Whether it's:

- A new data loader (Parquet, Avro, databases)
- Integration with another orchestration framework
- Framework-specific optimizations
- New export formats

Your contributions help everyone. Check out our [Contributing Guide](contributing/overview.md) to get started!

## Who's This For? 🎯

**You'll love MLPotion if you:**

- Switch between frameworks and hate rewriting everything
- Value heavily tested code that you can reuse
- Value type safety and IDE autocomplete (who doesn't?)
- Want production-ready code without enterprise bloat
- Believe ML pipelines should be composable and testable

**You might want something else if you:**

- Do not like modularity
- Do not like reusability
- Are to lazy to contribute something that you can't already find here

## Getting Started 🚀

Ready to start brewing? Here's your path:

<div class="getting-started-path">
  <div class="path-step">
    <div class="step-number">1</div>
    <div class="step-content">
      <h4>📥 Install MLPotion</h4>
      <p>Choose your framework flavor</p>
      <a href="getting-started/installation/">Installation Guide →</a>
    </div>
  </div>

  <div class="path-step">
    <div class="step-number">2</div>
    <div class="step-content">
      <h4>⚡ Quick Start</h4>
      <p>Get up and running in 5 minutes</p>
      <a href="getting-started/quickstart/">Quick Start →</a>
    </div>
  </div>

  <div class="path-step">
    <div class="step-number">3</div>
    <div class="step-content">
      <h4>🧠 Learn Concepts</h4>
      <p>Understand the architecture</p>
      <a href="getting-started/concepts/">Core Concepts →</a>
    </div>
  </div>

  <div class="path-step">
    <div class="step-number">4</div>
    <div class="step-content">
      <h4>🎨 Build Pipelines</h4>
      <p>Create your first pipeline</p>
      <a href="tutorials/basic-pipeline/">First Pipeline →</a>
    </div>
  </div>
</div>

## Show Me the Code! 💻

### Standalone Usage (Framework-Only)

=== "Keras"
    ```python linenums="1"
    --8<-- "docs/examples/keras/standalone.py"
    ```
=== "TensorFlow"
    ```python linenums="1"
    --8<-- "docs/examples/tensorflow/standalone.py"
    ```
=== "PyTorch"
    ```python linenums="1"
    --8<-- "docs/examples/pytorch/standalone.py"
    ```


### ZenML Pipeline Examples (MLOps Mode)

=== "Keras"
    ```python linenums="1"
    --8<-- "docs/examples/keras/zenml_pipeline.py"
    ```
=== "TensorFlow"
    ```python linenums="1"
    --8<-- "docs/examples/tensorflow/zenml_pipeline.py"
    ```
=== "PyTorch"
    ```python linenums="1"
    --8<-- "docs/examples/pytorch/zenml_pipeline.py"
    ```

## Feature Comparison 📊

| Feature | MLPotion | Framework-Only | All-in-One Solutions |
|---------|----------|----------------|---------------------|
| Multi-framework | ✅ Yes | ❌ No | ⚠️ Limited |
| Type Safety | ✅ Full | ⚠️ Partial | ⚠️ Partial |
| Modular Install | ✅ Yes | ❌ No | ❌ No |
| ZenML Native | ✅ Yes | ❌ Manual | ⚠️ Adapters |
| Learning Curve | 📈 Gentle | 📈 Framework-specific | 📈 Steep |
| Production Ready | ✅ Yes | ⚠️ DIY | ✅ Yes |
| Flexibility | 🌟🌟🌟🌟🌟 | 🌟🌟🌟🌟🌟 | 🌟🌟 |

## Community & Support 🤝

<div class="community-grid">
  <a href="https://github.com/UnicoLab/MLPotion" class="community-card">
    <div class="community-icon">🐙</div>
    <div class="community-title">GitHub</div>
    <div class="community-desc">Star, fork, contribute!</div>
  </a>

  <a href="https://github.com/UnicoLab/MLPotion/issues" class="community-card">
    <div class="community-icon">🐛</div>
    <div class="community-title">Issues</div>
    <div class="community-desc">Report bugs, request features</div>
  </a>

  <a href="https://unicolab.ai" class="community-card">
    <div class="community-icon">🦄</div>
    <div class="community-title">UnicoLab</div>
    <div class="community-desc">Enterprise AI solutions</div>
  </a>

  <div class="community-card">
    <div class="community-icon">📜</div>
    <div class="community-title">MIT License</div>
    <div class="community-desc">Free and open source</div>
  </div>
</div>

## What's Next? 🗺️

<div class="next-steps">
  <div class="next-step-card">
    <h3>📚 Learn the Basics</h3>
    <p>New to MLPotion? Start here!</p>
    <ul>
      <li><a href="getting-started/installation/">Installation</a></li>
      <li><a href="getting-started/quickstart/">Quick Start</a></li>
      <li><a href="getting-started/concepts/">Core Concepts</a></li>
    </ul>
  </div>

  <div class="next-step-card">
    <h3>🔧 Framework Guides</h3>
    <p>Deep dive into your framework</p>
    <ul>
      <li><a href="frameworks/tensorflow/">TensorFlow Guide</a></li>
      <li><a href="frameworks/pytorch/">PyTorch Guide</a></li>
      <li><a href="frameworks/keras/">Keras Guide</a></li>
    </ul>
  </div>

  <div class="next-step-card">
    <h3>🎓 Tutorials</h3>
    <p>Learn by building</p>
    <ul>
      <li><a href="tutorials/basic-pipeline/">Your First Pipeline</a></li>
      <li><a href="tutorials/zenml-integration/">ZenML Integration</a></li>
      <li><a href="tutorials/multi-framework/">Multi-Framework Project</a></li>
    </ul>
  </div>

  <div class="next-step-card">
    <h3>📖 API Reference</h3>
    <p>Detailed documentation</p>
    <ul>
      <li><a href="api/core/">Core APIs</a></li>
      <li><a href="api/frameworks/">Framework APIs</a></li>
      <li><a href="api/integrations/">Integrations</a></li>
    </ul>
  </div>
</div>

---

<p align="center">
  <strong>Ready to brew some ML magic? Let's get started! 🧪✨</strong><br>
  <em>Built with ❤️ for the ML community by <a href="https://unicolab.ai">🦄 UnicoLab</a></em>
</p>

<style>
/* Grid layouts */
.grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.grid-item {
  padding: 20px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 10px;
  border-left: 4px solid #4a86e8;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.grid-item:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 20px rgba(74, 134, 232, 0.2);
}

.grid-item h3 {
  margin-top: 0;
  color: #2c3e50;
}

.feature-list {
  list-style: none;
  padding-left: 0;
}

.feature-list li:before {
  content: "✓ ";
  color: #4CAF50;
  font-weight: bold;
  margin-right: 8px;
}

/* Getting started path */
.getting-started-path {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin: 30px 0;
}

.path-step {
  display: flex;
  align-items: flex-start;
  padding: 20px;
  background: white;
  border-radius: 10px;
  border: 2px solid #e9ecef;
  transition: all 0.3s ease;
}

.path-step:hover {
  border-color: #4a86e8;
  box-shadow: 0 5px 15px rgba(74, 134, 232, 0.1);
}

.step-number {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #4a86e8 0%, #2196F3 100%);
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 1.2em;
  margin-right: 20px;
}

.step-content h4 {
  margin: 0 0 5px 0;
  color: #2c3e50;
}

.step-content p {
  margin: 0 0 10px 0;
  color: #666;
}

.step-content a {
  color: #4a86e8;
  text-decoration: none;
  font-weight: 500;
}

.step-content a:hover {
  text-decoration: underline;
}

/* Community grid */
.community-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.community-card {
  padding: 25px;
  text-align: center;
  background: white;
  border-radius: 10px;
  border: 2px solid #e9ecef;
  text-decoration: none;
  color: inherit;
  transition: all 0.3s ease;
}

.community-card:hover {
  transform: translateY(-5px);
  border-color: #4a86e8;
  box-shadow: 0 5px 15px rgba(74, 134, 232, 0.2);
}

.community-icon {
  font-size: 3em;
  margin-bottom: 10px;
}

.community-title {
  font-weight: bold;
  font-size: 1.1em;
  margin-bottom: 5px;
  color: #2c3e50;
}

.community-desc {
  font-size: 0.9em;
  color: #666;
}

/* Next steps */
.next-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.next-step-card {
  padding: 20px;
  background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
  border-radius: 10px;
  border-top: 4px solid #4a86e8;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.next-step-card h3 {
  margin-top: 0;
  color: #2c3e50;
}

.next-step-card p {
  color: #666;
  margin-bottom: 15px;
}

.next-step-card ul {
  list-style: none;
  padding-left: 0;
}

.next-step-card li {
  margin-bottom: 8px;
}

.next-step-card a {
  color: #4a86e8;
  text-decoration: none;
}

.next-step-card a:hover {
  text-decoration: underline;
}

/* Responsive */
@media (max-width: 768px) {
  .grid-container,
  .community-grid,
  .next-steps {
    grid-template-columns: 1fr;
  }
}
</style>
