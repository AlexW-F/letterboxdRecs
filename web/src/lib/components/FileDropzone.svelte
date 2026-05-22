<script lang="ts">
	let {
		label,
		accept = '.csv',
		file = $bindable<File | null>(null),
		hint = ''
	}: {
		label: string;
		accept?: string;
		file?: File | null;
		hint?: string;
	} = $props();

	let dragging = $state(false);

	function onFile(f: File | null | undefined) {
		if (!f) return;
		file = f;
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragging = false;
		onFile(e.dataTransfer?.files?.[0]);
	}

	function onChange(e: Event) {
		const input = e.target as HTMLInputElement;
		onFile(input.files?.[0]);
	}
</script>

<label
	class="block rounded-lg border-2 border-dashed cursor-pointer transition px-4 py-6 text-center
		{dragging ? 'border-emerald-400 bg-emerald-500/5' : 'border-white/15 hover:border-white/30'}
		{file ? 'bg-emerald-500/10 border-emerald-400/60' : ''}"
	ondragover={(e) => {
		e.preventDefault();
		dragging = true;
	}}
	ondragleave={() => (dragging = false)}
	ondrop={onDrop}
>
	<div class="text-sm font-medium">{label}</div>
	{#if file}
		<div class="mt-1 text-xs text-emerald-300/90">{file.name} ({(file.size / 1024).toFixed(0)} KB)</div>
	{:else}
		<div class="mt-1 text-xs text-white/40">{hint || 'drop a CSV or click to choose'}</div>
	{/if}
	<input type="file" {accept} class="hidden" onchange={onChange} />
</label>
