<script lang="ts">
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import {
		UserPlus,
		Users as UsersIcon,
		Loader2,
		Link as LinkIcon,
		QrCode,
		ArrowRight,
		AtSign,
		FileText,
		Rss,
		CheckCircle2,
		ArrowLeft
	} from 'lucide-svelte';
	import FileDropzone from '$lib/components/FileDropzone.svelte';
	import Avatar from '$lib/components/Avatar.svelte';
	import QRCode from 'qrcode';
	import {
		getGroup,
		joinGroup,
		uploadLetterboxd,
		uploadLetterboxdUsername,
		type GroupState,
		type UploadResult
	} from '$lib/api';
	import { addMember } from '$lib/store';

	// The route guarantees an id; `?? ''` narrows the type so downstream
	// callers (joinGroup, encodeURIComponent) take it as a plain string.
	const groupId = $derived(page.params.id ?? '');
	let group = $state<GroupState | null>(null);
	let loadErr = $state<string | null>(null);

	let mode = $state<'username' | 'csv'>('csv');
	let name = $state('');
	let username = $state('');
	let ratingsFile = $state<File | null>(null);
	let watchedFile = $state<File | null>(null);
	let watchlistFile = $state<File | null>(null);
	let uploading = $state(false);
	let error = $state<string | null>(null);
	let copied = $state(false);
	// Name of the member added from this device — powers the "you're in"
	// confirmation so joining doesn't end on a silently-reset form.
	let joinedAs = $state<string | null>(null);

	const shareUrl = $derived(
		typeof window !== 'undefined' && groupId ? `${window.location.origin}/group/${groupId}/join` : ''
	);
	let qrDataUrl = $state<string>('');
	let qrOpen = $state(false);

	$effect(() => {
		if (!groupId) return;
		// Initial load surfaces errors; subsequent polls refresh silently so
		// members joining from other phones appear live without a manual reload.
		getGroup(groupId)
			.then((g) => (group = g))
			.catch((e) => (loadErr = e instanceof Error ? e.message : String(e)));
		const id = setInterval(() => {
			getGroup(groupId)
				.then((g) => (group = g))
				.catch(() => {});
		}, 6000);
		return () => clearInterval(id);
	});

	$effect(() => {
		if (!shareUrl) return;
		QRCode.toDataURL(shareUrl, {
			margin: 2,
			width: 280,
			color: { dark: '#0a0c10', light: '#ffffff' }
		})
			.then((url: string) => (qrDataUrl = url))
			.catch(() => (qrDataUrl = ''));
	});

	async function copyLink() {
		if (!shareUrl) return;
		try {
			await navigator.clipboard.writeText(shareUrl);
			copied = true;
			setTimeout(() => (copied = false), 1500);
		} catch {
			/* ignore */
		}
	}

	async function addMe() {
		error = null;
		if (!group) return;
		let memberName = name.trim();
		let uploadHash = '';

		uploading = true;
		try {
			let upload: UploadResult;
			if (mode === 'username') {
				const u = username.trim().replace(/^@/, '');
				if (!u) {
					error = 'Enter a Letterboxd username.';
					return;
				}
				upload = await uploadLetterboxdUsername(u);
				uploadHash = upload.hash;
				memberName = memberName || u;
			} else {
				if (!memberName) {
					error = 'Give yourself a name first.';
					return;
				}
				if (!ratingsFile) {
					error = 'Drop in a ratings.csv from your Letterboxd export.';
					return;
				}
				upload = await uploadLetterboxd(ratingsFile, watchedFile, watchlistFile);
				uploadHash = upload.hash;
			}
			const updated = await joinGroup(groupId, { name: memberName, hash: uploadHash });
			group = updated;
			// Persist this member to the device store so /explore + /group/* can use them.
			addMember({ name: memberName, hash: uploadHash, upload });
			joinedAs = memberName;
			name = '';
			username = '';
			ratingsFile = null;
			watchedFile = null;
			watchlistFile = null;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			uploading = false;
		}
	}
</script>

<svelte:head>
	<title>Join the group · movienight</title>
</svelte:head>

<section class="space-y-6 anim-fade-up max-w-3xl mx-auto">
	<header class="space-y-2">
		<span class="chip chip-violet">
			<UsersIcon size={11} />
			Shared group · {groupId}
		</span>
		<h1 class="display-md">
			Join the group.
		</h1>
		<p class="text-sm" style="color: var(--ink-muted);">
			Each friend who opens this link adds their own Letterboxd to the group — no need for one
			person to gather everyone's CSVs.
		</p>
	</header>

	{#if loadErr}
		<div class="surface p-4 text-sm" style="background: var(--rose-dim); border-color: rgba(187, 119, 68, 0.4); color: #dda679;">
			Could not load group <code>{groupId}</code>: {loadErr}
		</div>
	{:else if !group}
		<div class="surface p-5 text-sm" style="color: var(--ink-muted);">
			<Loader2 size={14} class="animate-spin inline mr-2" />
			Loading group…
		</div>
	{:else}
		<!-- Share link + QR -->
		<div class="surface p-4 space-y-3">
			<div class="flex items-center gap-3 flex-wrap">
				<LinkIcon size={14} style="color: var(--ink-faint);" />
				<code class="text-xs flex-1 min-w-0 truncate" style="color: var(--ink-muted);">{shareUrl}</code>
				<button
					class="btn btn-ghost btn-pill text-xs"
					onclick={() => (qrOpen = !qrOpen)}
					disabled={!qrDataUrl}
					title="Show QR code for phones to scan"
				>
					<QrCode size={12} />
					{qrOpen ? 'hide qr' : 'qr'}
				</button>
				<button class="btn btn-ghost btn-pill text-xs" onclick={copyLink}>
					{#if copied}
						<CheckCircle2 size={12} />
						copied
					{:else}
						copy
					{/if}
				</button>
			</div>
			{#if qrOpen && qrDataUrl}
				<div class="flex flex-col items-center gap-2 pt-1">
					<img
						src={qrDataUrl}
						alt="QR code for the group join link"
						class="rounded-md"
						style="background: white; padding: 4px;"
						width="280"
						height="280"
					/>
					<p class="text-[11px]" style="color: var(--ink-faint);">
						Have your friends scan this with their phone camera to join.
					</p>
				</div>
			{/if}
		</div>

		<!-- Existing members -->
		<div class="surface p-4 space-y-3">
			<div class="flex items-center gap-2">
				<UsersIcon size={14} style="color: var(--violet);" />
				<h2 class="text-sm font-medium">In the group ({group.members.length})</h2>
			</div>
			{#if group.members.length === 0}
				<p class="text-xs" style="color: var(--ink-faint);">
					No one's joined yet. Add yourself below to get this group started.
				</p>
			{:else}
				<ul class="space-y-2">
					{#each group.members as m (m.name)}
						<li class="flex items-center gap-3 text-sm">
							<Avatar name={m.name} size={28} />
							<div class="flex-1 min-w-0">
								<div class="font-medium leading-tight">{m.name}</div>
								<div class="text-[11px]" style="color: var(--ink-faint);">
									{m.n_ratings_mapped} ratings
									{#if m.n_watchlist > 0} · {m.n_watchlist} on watchlist{/if}
									{#if m.source === 'letterboxd_rss' && m.letterboxd_username}
										· @{m.letterboxd_username}
									{/if}
								</div>
							</div>
						</li>
					{/each}
				</ul>
			{/if}
		</div>

		<!-- Joined confirmation -->
		{#if joinedAs}
			<div
				class="surface p-4 flex items-center gap-3 flex-wrap anim-fade-up"
				style="background: rgba(143, 175, 122, 0.1); border-color: rgba(143, 175, 122, 0.45);"
			>
				<CheckCircle2 size={18} style="color: var(--green); flex-shrink: 0;" />
				<div class="flex-1 min-w-0 text-sm">
					<strong>You're in as {joinedAs}.</strong>
					{#if group.members.length >= 2}
						<span style="color: var(--ink-muted);">The group's ready when you are.</span>
					{:else}
						<span style="color: var(--ink-muted);">
							Share the link above — group recs unlock at 2 members.
						</span>
					{/if}
				</div>
				{#if group.members.length >= 2}
					<button
						class="btn btn-primary"
						onclick={() => goto(`/group/recommendations?group=${encodeURIComponent(groupId)}`)}
					>
						See group recs
						<ArrowRight size={14} />
					</button>
				{/if}
			</div>
		{/if}

		<!-- Add-me form -->
		<div class="surface p-5 space-y-4">
			<div class="flex items-center gap-2">
				<UserPlus size={16} style="color: var(--brand);" />
				<h2 class="text-base font-medium">{joinedAs ? 'Add someone else' : 'Add yourself'}</h2>
			</div>

			<div class="flex items-center gap-1 text-sm flex-wrap" role="radiogroup" aria-label="How to add your ratings">
				<button
					type="button"
					role="radio"
					aria-checked={mode === 'csv'}
					class="btn btn-pill text-xs {mode === 'csv' ? 'btn-secondary' : 'btn-ghost'}"
					onclick={() => (mode = 'csv')}
					disabled={uploading}
				>
					<FileText size={12} />
					CSV export
					<span class="text-[10px] mono ml-1" style="color: var(--brand);">full history</span>
				</button>
				<button
					type="button"
					role="radio"
					aria-checked={mode === 'username'}
					class="btn btn-pill text-xs {mode === 'username' ? 'btn-secondary' : 'btn-ghost'}"
					onclick={() => (mode = 'username')}
					disabled={uploading}
				>
					<AtSign size={12} />
					Letterboxd username
					<span class="text-[10px] mono ml-1" style="color: var(--rust);">~50 only</span>
				</button>
			</div>

			{#if mode === 'username'}
				<div class="space-y-2">
					<div class="flex items-stretch gap-0">
						<span
							class="inline-flex items-center px-3 rounded-l-md mono text-sm"
							style="background: rgba(215, 196, 131, 0.05); border: 1px solid var(--border); border-right: none; color: var(--ink-faint);"
						>
							letterboxd.com/
						</span>
						<input
							type="text"
							bind:value={username}
							placeholder="username"
							class="input"
							style="border-radius: 0 0.5rem 0.5rem 0;"
							autocomplete="off"
							spellcheck="false"
							disabled={uploading}
							onkeydown={(e) => { if (e.key === 'Enter' && !uploading) addMe(); }}
						/>
					</div>
					<input
						type="text"
						bind:value={name}
						placeholder="Display name (defaults to your username)"
						class="input"
						disabled={uploading}
					/>
					<div
						class="text-[11px] rounded-md px-3 py-2 leading-relaxed flex items-start gap-2"
						style="background: rgba(187, 119, 68, 0.1); border: 1px solid rgba(187, 119, 68, 0.4); color: #dda679;"
					>
						<Rss size={12} style="flex-shrink: 0; margin-top: 1px;" />
						<div>
							<strong>Heads up:</strong> usernames only pull your ~50 most-recent ratings via
							Letterboxd's RSS feed. Recommendations will be noticeably weaker than uploading
							your full <code>ratings.csv</code> export — switch to the CSV tab if you can.
						</div>
					</div>
				</div>
			{:else}
				<input
					type="text"
					bind:value={name}
					placeholder="Your display name"
					class="input"
					disabled={uploading}
				/>
				<FileDropzone label="ratings.csv" hint="from your Letterboxd export" bind:file={ratingsFile} />
				<FileDropzone label="watched.csv (optional)" hint="excludes already-watched + implicit signal" bind:file={watchedFile} />
				<FileDropzone label="watchlist.csv (optional)" hint="powers shared-watchlist overlap" bind:file={watchlistFile} />
				<div
					class="text-[11px] rounded-md px-3 py-2 leading-relaxed"
					style="background: rgba(215, 196, 131, 0.04); border: 1px solid var(--border); color: var(--ink-muted);"
				>
					<div class="font-medium" style="color: var(--ink-dim);">Where to find your CSVs</div>
					<div class="mt-1">
						On letterboxd.com: <strong>your profile → Settings → Data → Export your data</strong>.
						Unzip the download — the dropzones above accept the
						<code>ratings.csv</code>, <code>watched.csv</code>, and <code>watchlist.csv</code>
						files directly.
					</div>
					<a
						href="https://letterboxd.com/settings/data/"
						target="_blank"
						rel="noopener noreferrer"
						class="inline-block mt-1.5 underline"
						style="color: var(--brand);"
					>
						open Letterboxd data settings →
					</a>
				</div>
			{/if}

			{#if error}
				<p class="text-sm rounded-md px-3 py-2" style="background: var(--rose-dim); border: 1px solid rgba(187, 119, 68, 0.4); color: #dda679;">
					{error}
				</p>
			{/if}

			<button class="btn btn-primary w-full" onclick={addMe} disabled={uploading}>
				{#if uploading}
					<Loader2 size={14} class="animate-spin" />
					{mode === 'username' ? 'Fetching from Letterboxd…' : 'Uploading + enriching…'}
				{:else}
					<UserPlus size={14} />
					{joinedAs ? 'Add to the group' : 'Add me to the group'}
				{/if}
			</button>
		</div>

		<div class="flex justify-between items-center gap-3 flex-wrap">
			<button class="btn btn-ghost btn-pill text-xs" onclick={() => goto('/')}>
				<ArrowLeft size={12} />
				home
			</button>
			{#if group.members.length >= 2}
				<button
					class="btn btn-primary"
					onclick={() => goto(`/group/recommendations?group=${encodeURIComponent(groupId)}`)}
				>
					See group recs
					<ArrowRight size={14} />
				</button>
			{:else if group.members.length === 1}
				<span class="flex items-center gap-2 text-xs" style="color: var(--ink-muted);">
					<Loader2 size={12} class="animate-spin" />
					Waiting for friends to join… share the link above. Group recs unlock at 2 members.
				</span>
			{/if}
		</div>
	{/if}
</section>
