#include <pycuda-complex.hpp>
#include <stdio.h>
#define PI 3.141592654

typedef pycuda::complex<double> dcmplx;

__device__
dcmplx matrix_element(
    const dcmplx *A, 
    const dcmplx *B,
    uint a_offset,
    uint b_offset,
    uint row,
    uint col
)
{
    dcmplx value = 0.0;
    for (uint i= 0; i < {{dim}}; ++i) {
        value += A[a_offset + row*{{dim}} + i]*B[b_offset + i*{{dim}} + col];
    }

    return value;
}

__device__
void matrix_dot_add_existing_diag_left(
    const dcmplx *A, 
    const dcmplx *B,
    dcmplx *R,
    uint a_offset,
    uint b_offset,
    uint r_offset,
    dcmplx factor
)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            R[r_offset + row*{{dim}} + col] += factor*A[a_offset + row*{{dim}} + row]*B[b_offset + row*{{dim}} + col];
        }
    }
}

__device__
void matrix_dot_add_existing_diag_right(
    const dcmplx *A, 
    const dcmplx *B,
    dcmplx *R,
    uint a_offset,
    uint b_offset,
    uint r_offset,
    dcmplx factor
)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            R[r_offset + row*{{dim}} + col] += factor*A[a_offset + row*{{dim}} + col]*B[b_offset + col*{{dim}} + col];
        }
    }
}


__device__
void matrix_dot_add_existing(
    const dcmplx *A, 
    const dcmplx *B,
    dcmplx *R,
    uint a_offset,
    uint b_offset,
    uint r_offset,
    dcmplx factor
)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            R[r_offset + row*{{dim}} + col] += factor*matrix_element(A, B, a_offset, b_offset, row, col);
        }
    }
}

__device__
void matrix_dot(
    const dcmplx *A, 
    const dcmplx *B,
    dcmplx *R,
    uint a_offset,
    uint b_offset,
    uint r_offset,
    dcmplx factor
)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            R[r_offset + row*{{dim}} + col] = factor*matrix_element(A, B, a_offset, b_offset, row, col);
        }
    }
}

__device__
dcmplx matrix_trace(
    const dcmplx *A, 
    const dcmplx *B,
    uint a_offset,
    uint b_offset
)
{
    dcmplx result = 0;
    for (uint row=0; row < {{dim}}; ++row) {
        result += matrix_element(A, B, a_offset, b_offset, row, row);
    }

    return result;
}

__device__
void matrix_add(
    const dcmplx *A, 
    const dcmplx *B,
    dcmplx *R,
    uint a_offset,
    uint b_offset,
    uint r_offset,
    dcmplx a_factor,
    dcmplx b_factor
)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            R[r_offset + row*{{dim}} + col] = a_factor*A[a_offset + row*{{dim}} + col] + b_factor*B[b_offset + row*{{dim}} + col];
        }
    }
}

__device__
void matrix_add_3(
    const dcmplx *A, 
    const dcmplx *B,
    const dcmplx *C,
    dcmplx *R,
    uint a_offset,
    uint b_offset,
    uint c_offset,
    uint r_offset,
    dcmplx factor_a,
    dcmplx factor_b,
    dcmplx factor_c
)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            R[r_offset + row*{{dim}} + col] = factor_a*A[a_offset + row*{{dim}} + col] + factor_b*B[b_offset + row*{{dim}} + col] + factor_c*C[c_offset + row*{{dim}} + col];
        }
    }
}

__device__
void matrix_add_3_add_existing(
    const dcmplx *A, 
    const dcmplx *B,
    const dcmplx *C,
    dcmplx *R,
    uint a_offset,
    uint b_offset,
    uint c_offset,
    uint r_offset,
    dcmplx factor_a,
    dcmplx factor_b,
    dcmplx factor_c
)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            R[r_offset + row*{{dim}} + col] += factor_a*A[a_offset + row*{{dim}} + col] + factor_b*B[b_offset + row*{{dim}} + col] + factor_c*C[c_offset + row*{{dim}} + col];
        }
    }
}

__device__
void clear_matrix(dcmplx *A, uint a_offset)
{
    for (uint row= 0; row < {{dim}}; ++row) {
        for (uint col= 0; col < {{dim}}; ++col) {
            A[a_offset + row*{{dim}} + col] = 0.0;
        }
    }
}

__device__
void apply_lindblad(
    const dcmplx *rho, 
    const dcmplx *c_ops,
    const dcmplx *c_ops_dagger,
    dcmplx *tmp,
    dcmplx *k,
    uint range_matrix_offset
)
{
    for (uint j=0; j < {{num_c_ops}}; j++) {
        clear_matrix(tmp, 0);
        matrix_dot(c_ops, rho, tmp, j*{{mat_size}}, range_matrix_offset, 0, 1.0);
        matrix_dot_add_existing(tmp, c_ops_dagger, k, 0, j*{{mat_size}}, 0, 1.0);

        clear_matrix(tmp, 0);
        matrix_dot(rho, c_ops_dagger, tmp, range_matrix_offset, j*{{mat_size}}, 0, 1.0);
        matrix_dot_add_existing(tmp, c_ops, k, 0, j*{{mat_size}}, 0, -0.5);

        clear_matrix(tmp, 0);
        matrix_dot(c_ops_dagger, c_ops, tmp, j*{{mat_size}}, j*{{mat_size}}, 0, 1.0);
        matrix_dot_add_existing(tmp, rho, k, 0, range_matrix_offset, 0, -0.5);
    }
}

__global__
void mesolve(
    {{h0_args}},
    {{ht_args}},
    const dcmplx *c_ops,
    const dcmplx *c_ops_dagger,
    const double *step_sizes,
    dcmplx *rho,
    {{params}},
    {% if range_params %}
    {{range_params}},
    {% endif %}
    {{e_ops}},
    {{expects}}
)
{
    uint range_1 = threadIdx.x;  //col
    uint range_2 = threadIdx.y;  //row

    //if (range_1 == 1){
    //    return;
    //}

    uint range_offset = range_2*{{range_1_dim}} + range_1;
    uint range_matrix_offset = range_offset*{{mat_size}};

    //printf("%f\n", wp[range_1]);
    //printf("%d\n", range_1);
    //printf("%d,%d\n", range_matrix_offset, range_offset);
    //printf("%f\n", step_sizes[range_offset]);

    dcmplx imag(0, 1);

    dcmplx k1[{{mat_size}}];
    dcmplx k2[{{mat_size}}];
    dcmplx k3[{{mat_size}}];

    dcmplx tmp[{{mat_size}}];

    double t = 0.0;
    for(uint i=0; i < {{steps}}; i++) {
        clear_matrix(k1, 0);
        clear_matrix(k2, 0);
        clear_matrix(k3, 0);
        clear_matrix(tmp, 0);


        t = step_sizes[range_offset]*i;

        //K1
        //commutator
        //matrix_dot_add_existing(h0, rho, k1, 0, range_matrix_offset, 0, 1.0*imag); // h0*p
        //matrix_dot_add_existing(rho, h0, k1, range_matrix_offset, 0, 0, -1.0*imag); // -p*h0
        {% for h0 in h0_commutators_k1 %}
            {{h0}}
        {% endfor %}
        {% for h0 in ht_commutators_k1 %}
            {{h0}}
        {% endfor %}
        //lindblad
        apply_lindblad(rho,c_ops, c_ops_dagger, tmp, k1, range_matrix_offset);


        t += 0.5*step_sizes[range_offset];

        //K2
        //clear_matrix(tmp, range_matrix_offset);
        matrix_add(rho, k1, tmp, range_matrix_offset, 0, 0, 1.0, 0.5*step_sizes[range_offset]);

        //matrix_dot_add_existing(h0, tmp, k2, 0, range_matrix_offset, range_matrix_offset, 1.0*imag);
        //matrix_dot_add_existing(tmp, h0, k2, range_matrix_offset, 0, range_matrix_offset, -1.0*imag);
        {% for h0 in h0_commutators_k2 %}
            {{h0}}
        {% endfor %}
        {% for h0 in ht_commutators_k2 %}
            {{h0}}
        {% endfor %}

        //lindblad
        apply_lindblad(rho,c_ops, c_ops_dagger, tmp, k2, range_matrix_offset);


        t += 0.5*step_sizes[range_offset];

        //K3
        //clear_matrix(tmp, range_matrix_offset);
        matrix_add_3(rho, k1, k2, tmp, range_matrix_offset, 0, 0, 0, 1.0, -step_sizes[range_offset], 2.0*step_sizes[range_offset]);

        //matrix_dot_add_existing(h0, tmp, k2, 0, range_matrix_offset, range_matrix_offset, 1.0*imag);
        //matrix_dot_add_existing(tmp, h0, k2, range_matrix_offset, 0, range_matrix_offset, -1.0*imag);
        {% for h0 in h0_commutators_k3 %}
            {{h0}}
        {% endfor %}
        {% for h0 in ht_commutators_k3 %}
            {{h0}}
        {% endfor %}

        //lindblad
        apply_lindblad(rho,c_ops, c_ops_dagger, tmp, k3, range_matrix_offset);


        matrix_add_3_add_existing(k1, k2, k3, rho, 0,0,0, range_matrix_offset, step_sizes[range_offset]/6.0, step_sizes[range_offset]*4.0/6.0, step_sizes[range_offset]/6.0);
    
        {% for e in expectations%}
            {{e}}
        {% endfor %}
    }
}