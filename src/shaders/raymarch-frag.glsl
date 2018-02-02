#version 300 es

precision highp float;

uniform float u_Width;
uniform float u_Height;
uniform float u_Time;

out vec4 out_Col;

const float PI = 3.14159265359;                       
const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 1000.0;
const float EPSILON = 0.0001;

//intersection
float opI(float d1, float d2) {
    return max(d1, d2);
}
//union
float opU(float d1, float d2) {
    return min(d1, d2);
}
//subtraction
float opS(float d1, float d2) {
    return max(d1, -d2);
}


// polynomial smooth min (k = 0.1);
float smin( float a, float b, float k )
{
    float h = clamp( 0.5 + 0.5 * (b - a) / k, 0.0, 1.0 );
    return mix( b, a, h ) - k * h * (1.0 - h);
}

// exponential smooth min (k = 32);
float smin2( float a, float b, float k )
{
    float res = exp( -k * a ) + exp( -k * b );
    return -log( res ) / k;
}

float opBlend2(float d1, float d2 ) {
    return smin2( d1, d2, 0.1);
}

float opBlend(float d1, float d2 ) {
    return smin( d1, d2, 0.1);
}

//Rotation matrix around the X axis.
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}
//Rotation matrix around the Y axis.
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

//Rotation matrix around the Z axis.
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

//SDF for Box/cube
float boxSDF(vec3 p, vec3 size) {
    vec3 d = abs(p) - (size / 2.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance;
}

float sdTorus( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

float sdHexPrism( vec3 p, vec2 h ) {
    vec3 q = abs(p);
    return max(q.z - h.y, max((q.x*0.866025+q.y*0.5),q.y)-h.x);
}

float opRep( vec3 p, vec3 c ) {
    vec3 q = mod(p, c) - 0.5 * c;
    return boxSDF( q, vec3(1.0, 1.0, 1.0));
} 

float opRep2( vec3 p, vec3 c ) {
    vec3 q = mod(p, c) - 0.5 * c;
    return sdHexPrism( q, vec2(1.0, 2.0));
} 

//SDF for sphere at origin
float sphereSDF(vec3 p, float r) {
    return length(p) - r;
}

float sdEllipsoid( in vec3 p, in vec3 r ) {
    return (length( p / r ) - 1.0) * min(min(r.x, r.y), r.z);
}

vec3 Tx(vec3 p, vec3 tx) {
    return (p - tx);
}

// SDF for an XY aligned cylinder centered at the origin with height h and radius r.
float cylinderSDF(vec3 p, float h, float r) {
    float inOutRadius = length(p.xy) - r;
    float inOutHeight = abs(p.z) - h/2.0;
    float insideDistance = min(max(inOutRadius, inOutHeight), 0.0);
    float outsideDistance = length(max(vec2(inOutRadius, inOutHeight), 0.0));
    
    return insideDistance + outsideDistance;
}

float sdPlane( vec3 p, vec4 n )
{
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa, ba)/dot(ba, ba), 0.0, 1.0 );
    return length( pa - ba * h ) - r;
}

float softshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    for( float t = mint; t < maxt; t++)
    {
        float h = 0.0;
        if( h < 0.001 )
            return 0.0;
        res = min( res, k * h / t );
        t += h;
    }
    return res;
}

float opCheapBend( vec3 p ) {
    float c = cos(20.0 * sin(u_Time) * p.y);
    float s = sin(20.0 * sin(u_Time) * p.y);
    mat2  m = mat2(c, -s, s, c);
    vec3  q = vec3(m * p.xy, p.z);
    
    float rad = 0.9;
    vec3 a2 = vec3(0.8, 1.4, -1.1);
    vec3 b2 = vec3(-1.5, 1.9, -1.1);
    return sdCapsule(q, a2, b2, rad);
}

// float softShadow(vec3 ro, vec3 rd, float k)
// {
//    float res = 1.0;
//    float t = 0.12;         
//    for (int i = 0; i < SHADOW_RAY_DEPTH; i++)
//    {
//       float h = scene(ro + rd * t);
//       res = min(res, k * h / t);
//       t += h;
//       if (t > 5.0) break; 
//    }
//    return clamp(res, 0.25, 1.0);
// }

// cosine based palette, 4 vec3 params
vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{
    return a + b*cos( 6.28318*(c*t+d) );
}

/**
 * SDF describing the scene.
 * Absolute value of the return value indicates the distance to the surface.
 * Sign indicates whether the point is inside or outside the surface,
 * negative indicating inside.
 */
float sceneSDF(vec3 samplePoint) {    
    // Slowly spin the whole scene
    vec3 st = samplePoint;
    //st = rotateY(u_Time / 2.0) * samplePoint;

    //plane
    vec3 s = Tx(st, vec3(0.0, -1.2, 0.0));
    float boxes = opRep(s, vec3(1.1, 0.0, 1.3));

    vec3 s2 = Tx(st, vec3(0.0, -3.0, -10.0));
    float boxesBack = opRep2(s2, vec3(2.0, 2.0, 0.0)); 

    //top of heart
    vec3 ellipseRadius = vec3(1.1, 2.2, 0.9);
    vec3 boxSize = vec3(2.0, 8.0, 5.0);
    vec3 txEll = Tx(st, vec3(0.0, 2.4, -0.7));

    float ell1 = sdEllipsoid(rotateZ(radians(50.0)) * txEll, ellipseRadius);
    
    vec3 boxS1 = Tx(st, vec3(-1.0, 1.5, 0.0));
    float box1 = boxSDF(boxS1, boxSize);
    float bound1 = opI(ell1, box1);

    float ell2 = sdEllipsoid(rotateZ(radians(-50.0)) * txEll, ellipseRadius);

    vec3 boxS2 = Tx(st, vec3(1.0, 1.5, 0.0));
    float box2 = boxSDF(boxS2, boxSize);
    float bound2 = opI(ell2, box2);
    
    float head = opU(bound1, bound2);
    
    //eyes
    vec3 eyeS1 = Tx(st, vec3(0.24, 2.1, 0.1));
    float leftEye = sphereSDF(eyeS1, 0.12);
    
    float rad1 = 0.07;
    vec3 e1 = vec3(0.2, 2.3, 0.2);
    vec3 f1 = vec3(0.4, 2.4, 0.2);
    float leftEyeBrow = sdCapsule(st, e1, f1, rad1);

    vec3 eyeS2 = Tx(st, vec3(-0.25, 2.1, 0.1));
    float rightEye = sphereSDF(eyeS2, 0.12);

    vec3 e2 = vec3(-0.2, 2.3, 0.2);
    vec3 f2 = vec3(-0.4, 2.3, 0.2);
    float rightEyeBrow = sdCapsule(st, e2, f2, rad1);
    
    //mouth 
    vec3 mouthMove = Tx(st, vec3(0.0, 1.78, 0.07));
    vec3 mouth1 = vec3(0.35, 0.2, 0.2);
    float mouthShape = sdEllipsoid(mouthMove, mouth1);
    
    float mr = 0.03;
    vec3 ma1 = vec3(0.4, 1.8, 0.25);
    vec3 mb1 = vec3(-0.4, 1.8, 0.25);
    float open = sdCapsule(st, ma1, mb1, mr);

    float mouth = opS(mouthShape, open);
    head = opU(head, mouth);
    head = opU(head, leftEyeBrow);
    head = opU(head, rightEyeBrow);
    head = opU(head, leftEye);
    head = opU(head, rightEye);

    //body of Tata
    ellipseRadius = vec3(0.8, 1.0, 0.7);
    vec3 bodyS1 = Tx(st, vec3(0.0, 0.8, -1.1));
    float torso = sdEllipsoid(bodyS1, ellipseRadius);

    //arms
    float rad = 0.33;
    
    vec3 aa1 = vec3(0.0, 1.3, -1.1);
    vec3 bb1 = vec3(1.2, 0.8, -1.1);
    float bi = sdCapsule(st, aa1, bb1, rad);

    vec3 aa2 = vec3(1.2, 0.8, -1.1);
    vec3 bb2 = vec3(0.7, 0.6, -1.1);
    float fore = sdCapsule(st, aa2, bb2, rad);

    vec3 a2 = vec3(0.3, 0.5, -1.1);
    vec3 b2 = vec3(-1.5 - sin(u_Time), 1.6 + sin(u_Time), -1.1);
    float arm2 = sdCapsule(st, a2, b2, rad);
    //float bended = opCheapBend(st);

    vec3 st2 = Tx(st, vec3(0.0, 2.0, -1.1));

    float body = opBlend(torso, bi);

    body = opBlend(body, arm2);

    body = opU(body, fore);

    //body = opBlend(body, arm2);


    //legs
    vec3 a3 = vec3(0.35, 0.0, -1.1);
    vec3 b3 = vec3(0.4, -0.6, -1.1);
    float leg1 = sdCapsule(st, a3, b3, rad);

    vec3 a4 = vec3(-0.35, 0.0, -1.1);
    vec3 b4 = vec3(-0.4, -0.6, -1.1);
    float leg2 = sdCapsule(st, a4, b4, rad);

    body = opBlend(leg1, body);
    body = opBlend(leg2, body);

    float bodyAndHead = opU(body, head);
    float boxed = opU(boxesBack, boxes);
    return opU(bodyAndHead, boxed);
}

/**
 * Return the shortest distance from the eye to the scene surface along
 * the marching direction. 
 * If no part of the surface is found between start and end,
 * return end.
 * 
 * eye: origin of the ray
 * marchingDirection: the normalized direction
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */

 // 3D Perlin Noise
//	Simplex 3D Noise 
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //  x0 = x0 - 0. + 0.0 * C 
  vec3 x1 = x0 - i1 + 1.0 * C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1. + 3.0 * C.xxx;

// Permutations
  i = mod(i, 289.0 ); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients
// ( N*N points uniformly over a square, mapped onto an octahedron.)
  float n_ = 1.0/7.0; // N=7
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection);
        if (dist < EPSILON) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}

//Using the gradient of the SDF, estimate the normal on the surface at point p.
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}

/**
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    //float vis = shadowSoft( p, normalize(lightPos-p), 0.0925, length(lightPos-p), 128.0);

    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (
        k_d * dotLN + k_s * pow(dotRV, alpha));
}

/**
 * Lighting via Phong illumination.
 * 
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 */
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 lightPos = vec3(2.0, 10.0, 6.0);
    vec3 lightIntensity = vec3(0.5, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  lightPos,
                                  lightIntensity);  
    return color;
}

/**
 * Return a transform matrix that will transform a ray from view space
 * to world coordinates, given the eye point, the camera target, and an up vector.
 *
 * This assumes that the center of the camera is aligned with the negative z axis in
 * view space when calculating the ray marching direction. See rayDirection.
 */
mat3 viewMatrix(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat3(s, u, -f);
}

/**
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

void main() {
    //screen resolution
    vec2 u_Resolution = vec2(u_Width, u_Height);
    //View from eye
	vec3 rayDir = rayDirection(45.0, u_Resolution.xy, gl_FragCoord.xy);
    //camera position
    vec3 eye = vec3(0.0, 1.0, 40.0);

    //return a transform matrix that will transform a ray from view space to world coordinates,
    //given eye point, camera target, an up vector
    mat3 viewToWorld = viewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    
    vec3 worldDir = viewToWorld * rayDir;

    //sun
    vec3 sunDir = normalize(vec3(0, 0.01, 1.0));  //Sun position
    float sunSize = 30.0; // Sun can exist b/t 0 and 30 degrees from sunDir
    float sunAngle = acos(dot(rayDir, sunDir)) * 360.0 / PI;

    // if(sunAngle <= sunSize) {
    //     out_Col = vec4(1.0);
    // }
    // else {
    //     out_Col = vec4(0.1);
    // }

    float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);
    
    // Didn't hit anything
    if (dist > MAX_DIST - EPSILON) {
        float s = pow(max(0.0, snoise(rayDir * 4e2)), 18.0);
        out_Col = vec4(0.8, 0.2, 0.3, 1.0);
    }
    
    // The closest point on the surface to the eyepoint along the view ray
    vec3 p = eye + dist * worldDir;
    
    // Use the surface normal as the ambient color of the material
    vec3 K_a = (estimateNormal(p) + vec3(1.0)) / 1.5;
    vec3 K_d = K_a;
    vec3 K_s = vec3(1.0, 1.0, 1.0);
    float shininess = 100.0;
    
    vec3 color = phongIllumination(K_a, K_d, K_s, shininess, p, eye);
    out_Col = vec4(0.2, 0.0, 0.0, 0.0) + vec4(color, 1.0);
}