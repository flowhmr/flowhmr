// Thin client for the FlowHMR demo server (tools/web/server.py).

async function request(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { msg = (await res.json()).error || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return res;
}

const json = (path, options) => request(path, options).then((r) => r.json());
const postJson = (path, body) => json(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

export const getInfo = () => json("/api/info");
export const getJobs = () => json("/api/jobs");
export const getJob = (id) => json(`/api/jobs/${id}`);
export const getMeta = (id) => json(`/api/jobs/${id}/meta`);
export const getVertices = (id) => request(`/api/jobs/${id}/vertices`).then((r) => r.arrayBuffer());
export const getFaces = (id) => request(`/api/jobs/${id}/faces`).then((r) => r.arrayBuffer());
export const selectPerson = (id, person) => postJson(`/api/jobs/${id}/select`, { person });
export const rerunJob = (id, params) => postJson(`/api/jobs/${id}/rerun`, params);

export const videoUrl = (id) => `/api/jobs/${id}/video`;
export const npzUrl = (id) => `/api/jobs/${id}/npz`;
export const fbxUrl = (id) => `/api/jobs/${id}/fbx`;
export const personThumbUrl = (id, i) => `/api/jobs/${id}/person/${i}.jpg`;

/** Upload a video with per-job parameters. Resolves to the new job id. */
export function uploadVideo(file, params, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", file);
    for (const [k, v] of Object.entries(params)) form.append(k, String(v));
    const xhr = new XMLHttpRequest();
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      let res = {};
      try { res = JSON.parse(xhr.responseText); } catch { /* noop */ }
      if (xhr.status === 200 && res.job_id) resolve(res.job_id);
      else reject(new Error(res.error || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error("Upload failed (network error)"));
    xhr.open("POST", "/api/upload");
    xhr.send(form);
  });
}
