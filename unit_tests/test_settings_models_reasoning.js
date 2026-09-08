/* Run with: node unit_tests/test_settings_models_reasoning.js */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

// settings.css makes .hidden !important, overriding the inline display toggle.
const template = fs.readFileSync("templates/partials/settings/models.html", "utf8");
for (const id of ["reasoning-effort-group", "reasoning-effort-refresh"]) {
    const tag = template.match(new RegExp(`<[^>]*id="${id}"[^>]*>`))[0];
    assert.ok(!/class="[^"]*\bhidden\b/.test(tag), `${id} must allow inline display`);
}

const elements = new Map();
function element(id) {
    if (!elements.has(id)) elements.set(id, {
        id, value: "", style: {}, disabled: false, options: [],
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

    // Changing the model resets the effort; unsupported providers explain the default.
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
    assert.equal(element("reasoning-effort-group").style.display, "block");

    // Unknown models use a manual input; discovery switches back to a dropdown.
    element("model-provider").value = "codex";
    element("model-name-param").value = "new-model";
    lookup = async () => ({reasoning_supported: true,
        reasoning_capabilities: {efforts: [], manual: true}});
    settings.resetReasoningEffort("custom_level");
    await settings.updateReasoningOptions();
    const manual = element("model-reasoning-effort-manual");
    assert.equal(manual.value, "custom_level");
    assert.equal(manual.disabled, false);
    assert.equal(select.disabled, true);
    assert.equal(element("reasoning-effort-label").htmlFor, manual.id);
    manual.value = "high";
    settings.changeReasoningEffort();
    assert.equal(settings._reasoningEffort, "high");
    lookup = async () => supported;
    await settings.updateReasoningOptions();
    assert.equal(select.value, "high");
    assert.equal(manual.disabled, true);
    assert.equal(select.disabled, false);

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
