import { createApp } from "vue";
import App from "./App.vue";
import "./assets/main.css"; // Import Tailwind entry
import { parameterTooltips, parameterTutorials } from "./data/parameterHelp";

const app = createApp(App);

// 提供全局参数帮助数据
app.provide("parameterTooltips", parameterTooltips);
app.provide("parameterTutorials", parameterTutorials);

// 提供全局参数帮助函数
app.provide("parameterHelp", {
  openTutorial: (id) => {
    console.log("全局 openTutorial 被调用:", id);
    // 这个函数会在 ParameterHelpManager 组件挂载后被覆盖
  },
  closeTutorial: () => {
    console.log("全局 closeTutorial 被调用");
    // 这个函数会在 ParameterHelpManager 组件挂载后被覆盖
  },
  getTooltip: (id) => {
    console.log("全局 getTooltip 被调用:", id);
    return parameterTooltips[id] || null;
  },
});

// 添加全局错误处理
app.config.errorHandler = (err, vm, info) => {
  console.error("Vue 错误:", err);
  console.error("组件:", vm);
  console.error("信息:", info);
};

app.mount("#app");
