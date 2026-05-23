<script lang="ts">
	import { goto } from '$app/navigation';
	import { flip } from 'svelte/animate';
	import { fly, slide } from 'svelte/transition';
	import {
		UserPlus,
		Users as UsersIcon,
		ArrowRight,
		Trash2,
		Sparkles,
		Loader2,
		FileText,
		ChevronDown
	} from 'lucide-svelte';
	import FileDropzone from '$lib/components/FileDropzone.svelte';
	import Avatar from '$lib/components/Avatar.svelte';
	import MovieSpaceHero from '$lib/components/MovieSpaceHero.svelte';
	import { uploadLetterboxd } from '$lib/api';
	import { loadMembers, saveMembers, clearMembers, type Member } from '$lib/store';

	let members = $state<Member[]>(loadMembers());

	let name = $state('');
	let ratingsFile = $state<File | null>(null);
	let watchedFile = $state<File | null>(null);
	let uploading = $state(false);
	let error = $state<string | null>(null);

	$effect(() => {
		saveMembers(members);
	});

	async function addMember() {
		error = null;
		if (!name.trim()) {
			error = 'Give this friend a name first.';
			return;
		}
		if (!ratingsFile) {
			error = 'Drop in a ratings.csv from your Letterboxd export.';
			return;
		}
		uploading = true;
		try {
			const upload = await uploadLetterboxd(ratingsFile, watchedFile);
			members = [...members, { name: name.trim(), hash: upload.hash, upload }];
			name = '';
			ratingsFile = null;
			watchedFile = null;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			uploading = false;
		}
	}

	function removeMember(idx: number) {
		members = members.filter((_, i) => i !== idx);
	}

	function clearAll() {
		clearMembers();
		members = [];
	}

	function scrollToForm() {
		document.getElementById('add-friend')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}
</script>

<!--
  Landing page is a galaxy backdrop with content layered on top. The
  Three.js canvas is fixed full-viewport; content blocks float above
  with `pointer-events: none` on non-interactive copy so drag-to-spin
  reaches the canvas through dead space.
-->

<div
	class="fixed inset-0 -z-10"
	style="pointer-events: auto; isolation: isolate;"
>
	<!-- Backdrop mode: no drag, gentle parallax. Just drift + watch. -->
	<MovieSpaceHero
		height="100vh"
		points={5000}
		enableDrag={false}
		parallaxStrength={0.25}
	/>
</div>

<!-- Top scrim so the hero text reads cleanly against the galaxy -->
<div
	class="fixed inset-x-0 top-0 -z-[5] pointer-events-none"
	style="height: 60vh; background: radial-gradient(ellipse at top center, rgba(10, 12, 16, 0.55) 0%, transparent 70%);"
></div>

<!-- pointer-events: none lets drag/hover fall through to the galaxy
     canvas where there isn't an interactive element. Surfaces + buttons
     re-enable pointer-events explicitly. -->
<section class="relative" style="z-index: 1; pointer-events: none;">
	<!-- Hero — text floats over the galaxy. pointer-events: none lets the
	     user drag/spin the canvas behind it without dead zones. -->
	<div
		class="anim-fade-up min-h-[78vh] flex flex-col items-center justify-center pb-12"
		style="pointer-events: none;"
	>
		<!-- Frosted glass card behind the hero text. Keeps the headline +
		     paragraph readable no matter what cluster passes behind. -->
		<div
			class="hero-card relative px-6 sm:px-10 py-7 sm:py-9 text-center space-y-5 max-w-3xl"
			style="
				background: radial-gradient(ellipse at center, rgba(10, 12, 16, 0.78) 30%, rgba(10, 12, 16, 0.55) 75%, rgba(10, 12, 16, 0) 100%);
				border-radius: 1.5rem;
				backdrop-filter: blur(10px) saturate(140%);
				-webkit-backdrop-filter: blur(10px) saturate(140%);
				box-shadow: 0 30px 80px -20px rgba(0, 0, 0, 0.55);
			"
		>
			<span class="chip chip-accent" style="pointer-events: auto;">
				<Sparkles size={11} />
				5,000 films · drifting through ALS latent space
			</span>
			<h1
				class="display-xl text-balance mx-auto"
				style="font-family: 'Instrument Serif', Georgia, serif; font-style: italic; max-width: 24ch;"
			>
				Movie recommendations <br />
				<span class="text-gradient">your whole group</span> will love.
			</h1>
			<p
				class="text-balance max-w-2xl mx-auto"
				style="color: var(--ink); font-size: 1.05rem; line-height: 1.55;"
			>
				Each friend uploads their Letterboxd export. The backend folds everyone into the latent
				space you're hovering through, blends in Tag Genome relevance + TMDB plot embeddings,
				then re-ranks for diversity, popularity-debiased semantic alignment, and group
				fairness.
			</p>
			<div class="flex flex-wrap items-center gap-3 justify-center" style="pointer-events: auto;">
				<button class="btn btn-primary" onclick={scrollToForm}>
					<UserPlus size={15} />
					Add a friend
				</button>
				<button class="btn btn-ghost" onclick={() => goto('/me')}>
					Just me →
				</button>
			</div>
		</div>
		<button
			type="button"
			class="text-xs flex items-center gap-1 opacity-70 hover:opacity-100 transition mt-8"
			style="color: var(--ink-dim); pointer-events: auto;"
			onclick={scrollToForm}
			aria-label="scroll to upload form"
		>
			<ChevronDown size={14} class="animate-bounce" />
			scroll · upload below
		</button>
	</div>

	<!-- Upload + group panel. Frosted-glass surface keeps content readable
	     over the galaxy. -->
	<div
		id="add-friend"
		class="grid grid-cols-1 md:grid-cols-5 gap-4 anim-fade-up scroll-mt-24"
		style="animation-delay: 80ms;"
	>
		<div
			class="surface p-5 md:col-span-3 space-y-4"
			style="background: rgba(10, 12, 16, 0.72); backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%); pointer-events: auto;"
		>
			<div class="flex items-center gap-2">
				<UserPlus size={18} style="color: var(--brand);" />
				<h2 class="text-lg font-medium">Add a friend</h2>
			</div>
			<input
				type="text"
				bind:value={name}
				placeholder="Their name (e.g. alex, sara, mike)"
				class="input"
				disabled={uploading}
			/>
			<FileDropzone
				label="ratings.csv"
				hint="from your Letterboxd export · TMDB enrichment happens server-side"
				bind:file={ratingsFile}
			/>
			<FileDropzone
				label="watched.csv (optional)"
				hint="excludes already-watched films + feeds ALS implicit signal"
				bind:file={watchedFile}
			/>
			{#if error}
				<p
					class="text-sm rounded-md px-3 py-2"
					style="background: var(--rose-dim); border: 1px solid rgba(248, 113, 113, 0.3); color: #fecaca;"
				>
					{error}
				</p>
			{/if}
			<button class="btn btn-primary w-full" onclick={addMember} disabled={uploading}>
				{#if uploading}
					<Loader2 size={16} class="animate-spin" />
					Uploading + enriching with TMDB…
				{:else}
					<UserPlus size={16} />
					Add to group
				{/if}
			</button>
			{#if uploading}
				<p in:slide={{ duration: 200 }} class="text-xs" style="color: var(--ink-dim);">
					First time around, the backend hits TMDB for each film (~15s for ~250 ratings). Cached
					after that — same file is instant on re-upload.
				</p>
			{/if}
		</div>

		<div
			class="md:col-span-2 surface p-5 flex flex-col"
			style="background: rgba(10, 12, 16, 0.72); backdrop-filter: blur(14px) saturate(140%); -webkit-backdrop-filter: blur(14px) saturate(140%); pointer-events: auto;"
		>
			<div class="flex items-center justify-between">
				<div class="flex items-center gap-2">
					<UsersIcon size={18} style="color: var(--violet);" />
					<h2 class="text-lg font-medium">Group</h2>
					<span class="chip chip-violet">{members.length}</span>
				</div>
				{#if members.length > 0}
					<button
						class="btn btn-ghost btn-pill text-xs px-3 py-1"
						onclick={clearAll}
					>
						<Trash2 size={12} />
						clear
					</button>
				{/if}
			</div>

			<div class="mt-4 flex-1 space-y-2 max-h-72 overflow-auto pr-1">
				{#if members.length === 0}
					<div
						class="border-2 border-dashed rounded-lg p-6 text-center"
						style="border-color: var(--border); color: var(--ink-dim);"
					>
						<FileText size={24} class="mx-auto mb-2 opacity-60" />
						<p class="text-sm">No friends yet.</p>
						<p class="text-xs mt-1" style="color: var(--ink-faint);">
							Add at least 2 to enable group recs.
						</p>
					</div>
				{/if}
				{#each members as m, i (m.hash + i)}
					<div
						class="flex items-center gap-3 px-2 py-2 rounded-lg card-hover surface"
						style="border-radius: 0.6rem; background: rgba(255, 255, 255, 0.04);"
						in:fly={{ y: -8, duration: 220 }}
						animate:flip={{ duration: 220 }}
					>
						<Avatar name={m.name} size={36} />
						<div class="flex-1 min-w-0">
							<div class="text-sm font-medium truncate">{m.name}</div>
							<div class="text-[10px] mono truncate" style="color: var(--ink-faint);">
								{m.upload.n_ratings_mapped}/{m.upload.n_ratings_in} mapped · {m.hash.slice(0, 8)}…
							</div>
						</div>
						<button
							class="btn btn-ghost text-xs p-1.5"
							onclick={() => removeMember(i)}
							aria-label="remove"
							style="border-radius: 999px;"
						>
							<Trash2 size={13} />
						</button>
					</div>
				{/each}
			</div>

			<div class="mt-4 space-y-2 pt-2 border-t" style="border-color: var(--border);">
				<button
					class="btn btn-secondary w-full"
					onclick={() => goto('/group/recommendations')}
					disabled={members.length < 2}
					title={members.length < 2 ? 'Need at least 2 members' : ''}
				>
					<Sparkles size={15} />
					Generate group recs
					{#if members.length >= 2}
						<ArrowRight size={14} />
					{/if}
				</button>
				<button
					class="btn btn-ghost w-full text-xs"
					onclick={() => goto('/me')}
				>
					Just me · solo recommendations →
				</button>
			</div>
		</div>
	</div>

	<!-- Pitch row underneath, also over the galaxy -->
	<div
		class="mt-10 grid grid-cols-1 md:grid-cols-3 gap-3"
		style="z-index: 1;"
	>
		{#each [
			{
				title: 'Six aggregation strategies',
				body: 'average · least_misery · most_pleasure · consensus · hybrid · and our group_taste_vector that fuses everyone into a super-user.',
				tint: 'chip-violet'
			},
			{
				title: 'Three content scorers',
				body: 'Tag Genome 2021 (1,084 curated tags) · TMDB plot sentence-embeddings · IMDb director one-hots. Per-scorer weights tunable.',
				tint: 'chip-brand'
			},
			{
				title: 'Personal 3D in /explore',
				body: 'Your folded-in taste vector projected into the same UMAP space, with your rated films visible around you.',
				tint: 'chip-accent'
			}
		] as item, i (item.title)}
			<div
				class="surface p-4 card-hover anim-fade-up"
				style="
					background: rgba(10, 12, 16, 0.65);
					backdrop-filter: blur(12px);
					-webkit-backdrop-filter: blur(12px);
					animation-delay: {120 + i * 50}ms;
					pointer-events: auto;
				"
			>
				<span class="chip {item.tint}">{item.title}</span>
				<p class="text-sm mt-3" style="color: var(--ink-muted); line-height: 1.5;">{item.body}</p>
			</div>
		{/each}
	</div>
</section>
