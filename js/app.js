import { initState } from './state.js';
import { renderApp } from './render.js';

async function main() {
  await initState();
  renderApp();
}

main().catch((error) => {
  console.error(error);
  const app = document.getElementById('app');
  if (app) app.textContent = `初始化失败：${error.message}`;
});
