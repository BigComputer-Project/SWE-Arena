import './index.css' // import css for tailwind

import { createApp, h } from 'vue'
import App from './App.vue'
// import router from './router'

const app = createApp({
    render: () => h('div', { class: 'w-full h-full p-0 m-0' }, [h(App)])
})

// app.use(router)

app.mount('#app')
