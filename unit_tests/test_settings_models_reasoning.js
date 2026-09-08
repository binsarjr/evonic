/* Run with: node unit_tests/test_settings_models_reasoning.js */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const elements = new Map();
function element(id) {
    if (!elements.has(id)) elements.set(id, {
        value: "", style: {}, disabled: false, options: [],
        replaceChildren(...options) { this.options = options; },
        add(option) { this.options.push(option); },
        setCustomValidity(message) { this.validationMessage = message; },
    });
    return elements.get(id);
}
let lookup;
const context = {
    window: {}, document: { getElementById: element }, URLSearchParams,
    Option: function (text, value) { this.text = text; this.value = value; },
    apiGet: (...args) => lookup(...args),
};
vm.runInNewContext(fs.readFileSync("static/js/settings-models.js", "utf8"), context);
const settings = context.window.settingsModels;
const supported = { reasoning_supported: true, can_refresh: true,
    reasoning_capabilities: { efforts: ["low", "high", "max"], default_effort: "high" } };

(async () => {
    element("model-provider").value = "deepseek";
    element("model-name-param").value = "deepseek-v4-pro";
    element("model-api-format").value = "openai";
    lookup = async () => supported;
    settings.resetReasoningEffort("max");
    await settings.updateReasoningOptions();
    const select = element("model-reasoning-effort");
    assert.equal(select.value, "max");
    assert.equal(select.options.length, 4);
    assert.equal(element("reasoning-effort-group").style.display, "block");
    assert.equal(select.validationMessage, "");

    // Changing the model resets the effort; an unsupported provider hides the field.
    element("model-name-param").value = "unknown";
    await settings.updateReasoningOptions();
    assert.equal(settings._reasoningEffort, null);
    element("model-provider").value = "gemini";
    lookup = async () => ({ reasoning_supported: false });
    element("model-base-url").value = "https://api.deepseek.com/v1";
    settings.providers = [{ id: "gemini", api_format: "openai" }];
    settings.changeModelProvider();
    await settings._reasoningLookup;
    assert.equal(element("model-base-url").value, "");
    assert.equal(element("model-api-format").value, "openai");
    assert.equal(element("reasoning-effort-group").style.display, "none");

    // A failed lookup must not silently discard a saved value.
    settings.resetReasoningEffort("max");
    lookup = async () => { throw new Error("offline"); };
    await settings.updateReasoningOptions();
    assert.equal(select.value, "max");
    assert.ok(select.validationMessage);
    select.value = "";
    settings.changeReasoningEffort();
    assert.equal(settings._reasoningEffort, null);
    assert.equal(select.validationMessage, "");

    // Late responses from another provider cannot overwrite current options.
    let finish;
    lookup = () => new Promise((resolve) => { finish = resolve; });
    const pending = settings.updateReasoningOptions();
    element("model-provider").value = "deepseek";
    lookup = async () => supported;
    await settings.updateReasoningOptions();
    finish({ reasoning_supported: false });
    await pending;
    assert.equal(select.options.length, 4);
    assert.equal(element("reasoning-effort-group").style.display, "block");
    console.log("Reasoning dropdown checks passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
