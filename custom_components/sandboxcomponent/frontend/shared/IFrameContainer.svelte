<script lang="ts">
  import { onMount } from "svelte";
  import type { UserInteraction } from "../types";

  export const elem_classes: string[] = [];
  export let value: string;
  export let visible = true;
  export let min_height = false;
  export let isDarkMode = false;
  export let updateUserInteractions: (newInteraction: UserInteraction) => void;

  function getCurrentTimeInISO() {
    return new Date().toISOString();
  }

  const SUPPORTED_EVENT_TYPES = ["load", "resize", "keydown", "click", "scroll", "captureError"];

  onMount(() => {
    // Get the iframe element and its contentWindow
    const iframeElement = document.getElementById("sandboxIframe") as HTMLIFrameElement;
    const expectedSource = iframeElement ? iframeElement.contentWindow : null;

    function handleMessage(event: MessageEvent) {
      // Only process messages coming from the expected iframe
      if (expectedSource && event.source !== expectedSource) {
        return;
      }

      const data = event.data;
      // console.log("Received message from iframe", data);
      // Only record if type is one of the supported types.
      if (typeof data !== "object" || !data.type|| !SUPPORTED_EVENT_TYPES.includes(data.type?.toLowerCase())) {
        return;
      }

      const { type, time, ...rest } = data;
      const finalTime = time ?? getCurrentTimeInISO();

      updateUserInteractions({
        type,
        time: finalTime,
        ...rest,
      });
    }
    // Listen for messages from the iframe
    window.addEventListener("message", handleMessage);

    // Cleanup on unmount
    return () => {
      window.removeEventListener("message", handleMessage);
    };
  });
</script>

<div
  class={`relative flex flex-col flex-1 w-full h-full 
    ${min_height ? "min" : ""}
    ${!visible ? "hidden" : ""}
    ${isDarkMode ? "bg-black text-white" : "bg-white text-black"}`}
>
  <iframe
    id="sandboxIframe"
    title="iframe component"
    src={value}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; display-capture"
    sandbox="allow-forms allow-pointer-lock allow-popups allow-same-origin allow-scripts"
    allowtransparency
    allowfullscreen
    class="w-full h-full flex-1"
  />
</div>