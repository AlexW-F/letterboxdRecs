// localStorage store for uploaded members so /explore and /group/* can find
// this device's uploads. Persisted across tabs/sessions (a movie night can
// easily outlive a mobile tab); only display names + content hashes are
// stored, never ratings. Reads fall back to the legacy sessionStorage key
// so in-flight tabs from before the switch keep their members.

import type { UploadResult } from './api';

export interface Member {
	name: string;
	hash: string;
	upload: UploadResult;
}

const KEY = 'lbrecs.members';

export function loadMembers(): Member[] {
	if (typeof window === 'undefined') return [];
	try {
		const raw = localStorage.getItem(KEY) ?? sessionStorage.getItem(KEY);
		return raw ? (JSON.parse(raw) as Member[]) : [];
	} catch {
		return [];
	}
}

export function saveMembers(members: Member[]): void {
	if (typeof window === 'undefined') return;
	try {
		localStorage.setItem(KEY, JSON.stringify(members));
		sessionStorage.removeItem(KEY);
	} catch {
		/* storage blocked or full — non-fatal, members just won't persist */
	}
}

/**
 * Insert or update a member by hash. Used by upload sites (solo + group join)
 * so /explore and /group/* can find the device's known uploads.
 */
export function addMember(member: Member): void {
	const existing = loadMembers().filter((m) => m.hash !== member.hash);
	existing.push(member);
	saveMembers(existing);
}

export function clearMembers(): void {
	if (typeof window === 'undefined') return;
	try {
		localStorage.removeItem(KEY);
		sessionStorage.removeItem(KEY);
	} catch {
		/* ignore */
	}
}
