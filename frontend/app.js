/**
 * ChoreoAI Real-Time Visualizer
 */

let scene, camera, renderer, clock;
let skeleton, joints = [];
let currentMotion = null;
let currentFrame = 0;
let isPlaying = false;

// MediaPipe / ChoreoAI skeleton mapping (17 joints)
const JOINT_NAMES = [
    "Hips", "LeftHip", "RightHip", "Spine", "LeftKnee", "RightKnee", 
    "Neck", "LeftAnkle", "RightAnkle", "LeftShoulder", "RightShoulder",
    "LeftElbow", "RightElbow", "LeftWrist", "RightWrist", "LeftToe", "RightToe"
];

const BONES = [
    [0, 1], [0, 2], [0, 3], // Hips to hips/spine
    [1, 4], [4, 7], [7, 15], // Left leg
    [2, 5], [5, 8], [8, 16], // Right leg
    [3, 6],                  // Spine to neck
    [6, 9], [9, 11], [11, 13], // Left arm
    [6, 10], [10, 12], [12, 14] // Right arm
];

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('container').appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(1, 1, 1).normalize();
    scene.add(directionalLight);

    camera.position.set(0, 1.5, 5);
    camera.lookAt(0, 1, 0);

    const grid = new THREE.GridHelper(10, 10, 0x00d4ff, 0x333333);
    scene.add(grid);

    // Create skeleton placeholders
    skeleton = new THREE.Group();
    for (let i = 0; i < 17; i++) {
        const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.05),
            new THREE.MeshPhongMaterial({ color: 0x00d4ff })
        );
        joints.push(sphere);
        skeleton.add(sphere);
    }
    
    // Create bones (lines)
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
    for (let i = 0; i < BONES.length; i++) {
        const geometry = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(), new THREE.Vector3()
        ]);
        const line = new THREE.Line(geometry, lineMaterial);
        skeleton.add(line);
    }

    scene.add(skeleton);

    window.addEventListener('resize', onWindowResize, false);
    
    document.getElementById('generate-btn').addEventListener('click', generateMotion);
    document.getElementById('play-btn').addEventListener('click', () => { isPlaying = true; });
    document.getElementById('pause-btn').addEventListener('click', () => { isPlaying = false; });
    document.getElementById('timeline').addEventListener('input', (e) => {
        currentFrame = parseInt(e.target.value);
        updateSkeleton();
    });

    animate();
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

async function generateMotion() {
    const prompt = document.getElementById('prompt-input').value;
    if (!prompt) return;

    document.getElementById('status').innerText = "Generating...";
    
    try {
        const response = await fetch('http://localhost:8000/generate_motion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, seq_len: 120 })
        });
        
        const data = await response.json();
        if (data.status === "success") {
            currentMotion = data.motion;
            currentFrame = 0;
            isPlaying = true;
            document.getElementById('timeline').max = currentMotion.length - 1;
            document.getElementById('status').innerText = "Generated successfully.";
        } else {
            document.getElementById('status').innerText = "Error: " + data.detail;
        }
    } catch (err) {
        document.getElementById('status').innerText = "Connection failed.";
        console.error(err);
    }
}

function updateSkeleton() {
    if (!currentMotion) return;

    const frame = currentMotion[currentFrame];
    // Update joint positions
    for (let i = 0; i < 17; i++) {
        const p = frame[i];
        joints[i].position.set(p[0], p[1], p[2]);
    }

    // Update bones
    let lineIdx = 0;
    skeleton.children.forEach(child => {
        if (child instanceof THREE.Line) {
            const pair = BONES[lineIdx++];
            const p1 = joints[pair[0]].position;
            const p2 = joints[pair[1]].position;
            child.geometry.setFromPoints([p1, p2]);
        }
    });

    document.getElementById('frame-counter').innerText = `Frame: ${currentFrame + 1}/${currentMotion.length}`;
    document.getElementById('timeline').value = currentFrame;
}

function animate() {
    requestAnimationFrame(animate);
    
    if (isPlaying && currentMotion) {
        currentFrame = (currentFrame + 1) % currentMotion.length;
        updateSkeleton();
    }
    
    renderer.render(scene, camera);
}

init();
